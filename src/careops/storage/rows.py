"""Store and query the same synthetic claim rows across five database engines."""

import json
import re
import sqlite3
from contextlib import closing
from time import perf_counter

ENGINES = ("sqlite", "postgres", "mysql", "mongo", "redis")


class SqlClaims:
    def __init__(self, connection, namespace, dialect):
        self.connection = connection
        self.table = namespace
        self.dialect = dialect
        self.marker = "?" if dialect == "sqlite" else "%s"

    def setup(self):
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(f"""
                CREATE TABLE {self.table} (
                    claim_id VARCHAR(32) PRIMARY KEY,
                    service_type VARCHAR(32) NOT NULL,
                    amount_idr BIGINT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            cursor.execute(f"""CREATE INDEX {self.table}_service_amount
                ON {self.table}(service_type, amount_idr DESC, claim_id DESC)
            """)
        self.connection.commit()

    def replace(self, rows):
        markers = ", ".join([self.marker] * 4)
        update = (
            "AS incoming ON DUPLICATE KEY UPDATE service_type = incoming.service_type, "
            "amount_idr = incoming.amount_idr, payload = incoming.payload"
            if self.dialect == "mysql" else
            "ON CONFLICT (claim_id) DO UPDATE SET service_type = excluded.service_type, "
            "amount_idr = excluded.amount_idr, payload = excluded.payload"
        )
        values = [(row["claim_id"], row["service_type"], row["amount_idr"], json.dumps(row))
                  for row in rows]
        with closing(self.connection.cursor()) as cursor:
            cursor.executemany(f"INSERT INTO {self.table} VALUES ({markers}) {update}", values)
        self.connection.commit()

    def fetch(self, clause, params=()):
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(clause, params)
            return cursor.fetchall()

    def count(self):
        return self.fetch(f"SELECT COUNT(*) FROM {self.table}")[0][0]

    def get(self, claim_id):
        rows = self.fetch(f"SELECT payload FROM {self.table} WHERE claim_id = {self.marker}",
                          (claim_id,))
        return json.loads(rows[0][0]) if rows else None

    def largest(self, service_type, minimum, limit):
        rows = self.fetch(f"""
            SELECT claim_id FROM {self.table}
            WHERE service_type = {self.marker} AND amount_idr >= {self.marker}
            ORDER BY amount_idr DESC, claim_id DESC LIMIT {self.marker}
        """, (service_type, minimum, limit))
        return [row[0] for row in rows]

    def version(self):
        query = "SELECT sqlite_version()" if self.dialect == "sqlite" else "SELECT VERSION()"
        return self.fetch(query)[0][0]

    def cleanup(self):
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.connection.commit()

    def close(self):
        self.connection.close()


class MongoClaims:
    def __init__(self, client, name):
        self.client = client
        self.collection = client.careops_lab[name]

    def setup(self):
        self.collection.create_index([("service_type", 1), ("amount_idr", -1), ("_id", -1)])

    def replace(self, rows):
        from pymongo import ReplaceOne

        self.collection.bulk_write([
            ReplaceOne({"_id": row["claim_id"]}, {**row, "_id": row["claim_id"]}, upsert=True)
            for row in rows
        ])

    def count(self):
        return self.collection.count_documents({})

    def get(self, claim_id):
        return self.collection.find_one({"_id": claim_id}, {"_id": False})

    def largest(self, service_type, minimum, limit):
        filt = {"service_type": service_type, "amount_idr": {"$gte": minimum}}
        matches = self.collection.find(filt, {"_id": True}).sort([
            ("amount_idr", -1), ("_id", -1),
        ]).limit(limit)
        return [row["_id"] for row in matches]

    def version(self):
        return self.client.server_info()["version"]

    def cleanup(self):
        self.collection.drop()

    def close(self):
        self.client.close()


class RedisClaims:
    def __init__(self, client, name):
        self.client = client
        self.rows = name + ":rows"
        self.amounts = name + ":amounts"

    def setup(self):
        self.client.ping()

    def replace(self, rows):
        with self.client.pipeline(transaction=True) as batch:
            batch.hset(self.rows, mapping={row["claim_id"]: json.dumps(row) for row in rows})
            batch.zadd(self.amounts, {row["claim_id"]: row["amount_idr"] for row in rows})
            batch.execute()

    def count(self):
        return self.client.hlen(self.rows)

    def get(self, claim_id):
        value = self.client.hget(self.rows, claim_id)
        return json.loads(value) if value is not None else None

    def largest(self, service_type, minimum, limit):
        ids = self.client.zrevrangebyscore(self.amounts, "+inf", minimum)
        if not ids:
            return []
        # The sorted set applies the amount range. The small lab filters service type after it.
        return [row["claim_id"] for row in map(json.loads, self.client.hmget(self.rows, ids))
                if row["service_type"] == service_type][:limit]

    def version(self):
        return self.client.info("server")["redis_version"]

    def cleanup(self):
        self.client.delete(self.rows, self.amounts)

    def close(self):
        self.client.close()


def connect(engine, directory, namespace):
    if not re.fullmatch(r"claim_lab_[a-f0-9]{12}", namespace):
        raise ValueError("Expected a generated lab namespace")
    if engine == "sqlite":
        return SqlClaims(sqlite3.connect(directory / "claims.sqlite3"), namespace, engine)
    if engine == "postgres":
        import psycopg

        connection = psycopg.connect(host="127.0.0.1", port=55432, dbname="careops_lab",
                                     user="careops", password="careops-demo", connect_timeout=5)
        return SqlClaims(connection, namespace, engine)
    if engine == "mysql":
        import pymysql

        connection = pymysql.connect(host="127.0.0.1", port=53306, database="careops_lab",
                                     user="careops", password="careops-demo", connect_timeout=5)
        return SqlClaims(connection, namespace, engine)
    if engine == "mongo":
        from pymongo import MongoClient

        return MongoClaims(MongoClient("127.0.0.1", 57017, serverSelectionTimeoutMS=5000),
                           namespace)
    if engine == "redis":
        from redis import Redis

        return RedisClaims(Redis(host="127.0.0.1", port=56379, decode_responses=True,
                                 socket_connect_timeout=5, socket_timeout=5), namespace)
    raise ValueError(f"Unknown database {engine}")


def expected_ids(rows, service_type, minimum, limit):
    ordered = sorted(rows, key=lambda row: (row["amount_idr"], row["claim_id"]), reverse=True)
    return [row["claim_id"] for row in ordered
            if row["service_type"] == service_type and row["amount_idr"] >= minimum][:limit]


def exercise(engine, directory, namespace, rows):
    original = rows[0]
    other = next(row for row in rows if row["service_type"] != original["service_type"])
    changed = {**original, "amount_idr": original["amount_idr"] + 1_000_000,
               "service_type": other["service_type"], "service_code": other["service_code"],
               "attachments": other["attachments"]}
    updated = [changed, *rows[1:]]
    queries = [(service, 100_000, 3) for service in sorted({row["service_type"] for row in rows})]
    queries.append((original["service_type"], 10**12, 3))
    expected = [expected_ids(updated, *query) for query in queries]
    opened = connect(engine, directory, namespace)

    try:
        with closing(opened) as store:
            store.setup()
            start = perf_counter()
            store.replace(rows)
            insert_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            store.replace(updated)
            replay_ms = (perf_counter() - start) * 1000

        with closing(connect(engine, directory, namespace)) as store:
            if store.count() != len(rows):
                raise RuntimeError(f"{engine}: repeat write changed fixture cardinality")
            if any(store.get(row["claim_id"]) != row for row in updated):
                raise RuntimeError(f"{engine}: stored row differs after reconnecting")
            if store.get("SYN-D-000") is not None:
                raise RuntimeError(f"{engine}: absent key returned a row")
            timings = []
            for _ in range(5):
                for args, target in zip(queries, expected, strict=True):
                    start = perf_counter()
                    found = store.largest(*args)
                    timings.append((perf_counter() - start) * 1000)
                    if found != target:
                        raise RuntimeError(f"{engine}: amount range, service filter or order differs")
            return {
                "server_version": store.version(),
                "rows": len(rows),
                "upsert_and_reopen": True,
                "missing_key_returns_none": True,
                "batch_insert_ms": round(insert_ms, 3),
                "batch_replay_ms": round(replay_ms, 3),
                "queries": [
                    {"service_type": service, "minimum_idr": low, "limit": limit,
                     "claim_ids": selected}
                    for (service, low, limit), selected in zip(queries, expected, strict=True)
                ],
            }, timings
    finally:
        with closing(connect(engine, directory, namespace)) as store:
            store.cleanup()

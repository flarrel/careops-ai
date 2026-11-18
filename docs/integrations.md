# Local integrations

The integration examples share the same FastAPI review and history endpoints as the main application. The C++ client summarizes recent outcomes. The PHP client requests and decides a synthetic review. Django, Flask, and Laravel each expose the same ten-run summary at `GET /summary`.

## Import an Excel workbook

Run from the repository root with Python 3.12.

```bash
python3 -m pip install -e '.[rag,agent,integrations]'
work=$(mktemp -d)
printf "%s\n" "$work"
python3 -m careops.integrations.cli export-xlsx "$work/claims.xlsx" \
  --claims integrations/examples/spreadsheet-demo.jsonl
python3 -m careops.integrations.cli import-xlsx "$work/claims.xlsx" "$work/claims.jsonl"
CAREOPS_CLAIMS_PATH="$work/claims.jsonl" CAREOPS_HISTORY_PATH="$work/runs.sqlite3" \
  python3 -m uvicorn careops.api:app --host 127.0.0.1 --port 8000
```

Save the printed directory path as `work` in any additional terminal used for these commands. The `Claims` worksheet accepts up to 500 synthetic development fixtures. Keep the template headers, use an Excel boolean in `synthetic`, use integer IDR amounts, and separate attachment codes with `|`. The importer rejects formulas, duplicate IDs, non-synthetic rows, and unexpected columns. It writes validated JSON Lines in the format used by the read-only agent. The fixture contents remain behind the normal approval step. Omit `--claims` on export to use the 20 default development claims.

## Preview interoperability contracts

The example EMR envelope normalizes to the same row as the spreadsheet fixture. The versioned intake contract accepts `his` or `emr` as the source. Its fields are limited to synthetic references, service codes, attachments, and an IDR amount.

```bash
python3 -m careops.integrations.cli normalize-intake integrations/examples/emr-intake.json "$work/emr.jsonl"
python3 -m careops.integrations.cli preview SYN-D-099 --claims "$work/emr.jsonl" > "$work/preview.json"
curl -fsSL https://hl7.org/fhir/R4/fhir.schema.json.zip -o "$work/fhir.schema.json.zip"
python3 integrations/check_fhir.py "$work/fhir.schema.json.zip" "$work/preview.json"
```

The preview produces a draft [FHIR R4 Claim](https://hl7.org/fhir/R4/claim.html) with demo references. The service code and IDR amount come from the fixture. If the shared checklist finds issues, the preview also produces an offline [Dataverse Web API](https://learn.microsoft.com/en-us/power-apps/developer/data-platform/webapi/create-entity-web-api) task request with a synthetic claim ID and checklist messages. A clear fixture yields no task request. The examples model contracts and run locally. A live HIS, EMR, Dynamics, or WhatsApp integration needs a separate authorized environment.

## Run the consumers

The C++ client uses C++17, libcurl, and the nlohmann JSON 3.12.0 single header.

```bash
mkdir -p "$work/include/nlohmann"
curl -fsSL https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp \
  -o "$work/include/nlohmann/json.hpp"
printf '%s  %s\n' aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63 \
  "$work/include/nlohmann/json.hpp" | shasum -a 256 -c -
c++ -std=c++17 -Wall -Wextra -I "$work/include" integrations/cpp/summary.cpp \
  -lcurl -o "$work/careops-summary"
"$work/careops-summary"
php integrations/php/review.php SYN-D-099 --allow
```

Run the next three commands in separate terminals. Laravel requires PHP 8.2 or later with cURL and Composer.

```bash
python3 -m flask --app integrations.python.flask_app run --host 127.0.0.1 --port 8002
python3 -m django runserver 127.0.0.1:8003 --settings integrations.python.django_app --noreload
```

```bash
cd integrations/php/laravel
composer install --no-plugins --no-scripts
mkdir -p bootstrap/cache storage/logs
php -S 127.0.0.1:8004 -t public public/index.php
```

Each client defaults to the API at `http://127.0.0.1:8000` and accepts `CAREOPS_API_URL`. Run `python3 integrations/check_consumers.py "$work/careops-summary"` from the repository root to verify PHP approval and denial, the C++ output, and both HTTP responses from each framework.

## Verify the browser workflow

Start the [Next.js client](application.md) against the API using the imported workbook. From the repository root, run the Gherkin scenarios with `python3 -m behave integrations/features`. They check approval, denial, invalid IDs, the imported amount, and history in Chrome. Set `CHROMEDRIVER` and `CHROME_BINARY` to reuse a local driver and browser. The browser test uses `http://127.0.0.1:3000` by default, configurable with `CAREOPS_WEB_URL`.

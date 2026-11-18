import type { NextRequest } from "next/server";

type Context = { params: Promise<{ path: string[] }> };

async function forward(request: NextRequest, context: Context) {
  const route = (await context.params).path.join("/");
  const reading = request.method === "GET";
  const allowed = reading
    ? route === "runs"
    : route === "answer" || route === "reviews" || /^reviews\/[a-f0-9]{32}\/decision$/.test(route);
  if (!allowed) return Response.json({ detail: "Route unavailable." }, { status: 404 });

  const origin = request.headers.get("origin");
  if (!reading && origin && origin !== "http://" + request.headers.get("host")) {
    return Response.json({ detail: "Origin rejected." }, { status: 403 });
  }
  if (!reading && request.headers.get("content-type")?.split(";")[0] !== "application/json") {
    return Response.json({ detail: "JSON request required." }, { status: 415 });
  }

  const root = process.env.CAREOPS_API_URL ?? "http://127.0.0.1:8000";
  try {
    const upstream = await fetch(new URL("/" + route + request.nextUrl.search, root), {
      method: request.method,
      headers: reading ? undefined : { "Content-Type": "application/json" },
      body: reading ? undefined : await request.text(),
      cache: "no-store",
      signal: AbortSignal.timeout(reading ? 4000 : 60000),
    });
    return Response.json(await upstream.json(), { status: upstream.status, headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ detail: "The local API is unavailable. Try again shortly." }, { status: 503 });
  }
}

export { forward as GET, forward as POST };

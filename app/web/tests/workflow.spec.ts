import { expect, test } from "@playwright/test";

test("renders an actual cited answer and an abstention", async ({ page }) => {
  await page.goto("/");
  const workflow = page.getByRole("region", { name: "Review workflow" });
  const input = page.getByRole("textbox", { name: "Policy question" });
  await input.fill("Siapa pengguna pedoman INA-CBG untuk metode pembayaran klaim menurut Permenkes 26/2021?");
  await page.getByRole("button", { name: "Find evidence" }).click();
  await expect(workflow.getByText("Cited answer")).toBeVisible();
  await expect(workflow.getByRole("link")).toHaveAttribute("href", /^https:\/\//);
  await expect(workflow.getByText(/PDF page 3/)).toBeVisible();
  await expect(page.getByRole("table").getByText("Cited answer").first()).toBeVisible();

  await input.fill("What is the weather today?");
  await page.getByRole("button", { name: "Find evidence" }).click();
  await expect(workflow.getByText("Abstained")).toBeVisible();
  await expect(workflow.getByRole("link")).toHaveCount(0);
});

test("handles denied and approved lookups on a narrow screen", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const workflow = page.getByRole("region", { name: "Review workflow" });
  await page.getByRole("button", { name: "Synthetic review" }).click();
  await page.getByRole("textbox", { name: "Synthetic claim ID" }).fill("abc");
  await page.getByRole("button", { name: "Request review" }).click();
  await expect(workflow.getByRole("alert")).toHaveText("Invalid request. Check claim_id.");
  await page.getByRole("textbox", { name: "Synthetic claim ID" }).fill("SYN-D-001");
  await page.getByRole("button", { name: "Request review" }).click();
  await expect(workflow.getByText("SYN-D-001")).toHaveCount(0);
  await page.getByRole("button", { name: "Decline lookup" }).click();
  await expect(workflow.getByText("Synthetic lookup declined.")).toBeVisible();
  await expect(workflow.getByText("SYN-D-001")).toHaveCount(0);

  await page.getByRole("textbox", { name: "Synthetic claim ID" }).fill("SYN-D-001");
  await page.getByRole("button", { name: "Request review" }).click();
  await page.getByRole("button", { name: "Allow lookup" }).click();
  await expect(workflow.getByText("Missing fixture attachment: invoice.")).toBeVisible();
  await expect(workflow.getByText("SYN-D-001", { exact: true })).toBeVisible();
  await expect(page.getByRole("table").getByText("Needs human follow-up").first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("proxy accepts only application routes and JSON from the browser origin", async ({ request }) => {
  expect((await request.get("/api/docs")).status()).toBe(404);
  expect((await request.post("/api/answer", {
    headers: { Origin: "http://example.org" }, data: { query: "example" },
  })).status()).toBe(403);
  expect((await request.post("/api/answer", { data: "example" })).status()).toBe(415);
});

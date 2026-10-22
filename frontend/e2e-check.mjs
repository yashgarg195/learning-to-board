// Manual end-to-end check against running dev servers (see README).
// Usage: CHROME_PATH=... node e2e-check.mjs
import { chromium } from "playwright";

const errors = [];
const browser = await chromium.launch(
  process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
);
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
console.log("title:", await page.title());

// Empty state before any run
console.log("empty state:", await page.getByText("No simulation loaded").isVisible());

// Configure a scenario
await page.selectOption("#aircraft", "a320");
await page.fill("#seed", "7");
await page.selectOption("#arrival", "groups");
await page.selectOption("#strategy", "steffen");

await page.getByRole("button", { name: /Run simulation/i }).click();
await page.waitForSelector("text=Cabin — A320", { timeout: 30000 });
const ticks = await page.locator("text=Total boarding").locator("..").innerText();
console.log("metrics:", ticks.replace(/\n/g, " "));

const seats = await page.locator("svg[role=img] rect").count();
console.log("cabin svg rects:", seats);

// Playback: step, speed, scrub
await page.getByRole("button", { name: /Step forward one tick/i }).click();
console.log("after step:", await page.locator("text=/t 001/").first().innerText());
await page.getByRole("button", { name: "4×" }).click();
await page.getByRole("button", { name: /Play replay/ }).click();
await page.waitForTimeout(1500);
await page.getByRole("button", { name: /Pause replay/ }).click();
const mid = await page.locator("text=/t \\d{3}/").first().innerText();
console.log("after 1.5s play at 4x:", mid);

// Scrub via the congestion trace
const trace = page.locator("svg[role=slider]");
const box = await trace.boundingBox();
await page.mouse.click(box.x + box.width * 0.5, box.y + box.height / 2);
console.log("after scrub to 50%:", await page.locator("text=/t \\d{3}/").first().innerText());

await page.screenshot({ path: "/tmp/shot-sim.png", fullPage: true });

// Compare strategies
await page.getByRole("button", { name: /^Compare$/ }).click();
await page.waitForSelector("table", { timeout: 60000 });
const rows = await page.locator("tbody tr").allInnerTexts();
console.log("compare rows:", rows.map((r) => r.replace(/\n/g, " ")));
await page.screenshot({ path: "/tmp/shot-compare.png", fullPage: true });

// Train
await page.getByRole("tab", { name: /Reinforcement learning/i }).click();
await page.fill("#ep", "400");
await page.fill("#rs", "1");
await page.fill("#sa", "ui_trained");
await page.getByRole("button", { name: /Start training/i }).click();
await page.waitForSelector("text=Evaluation (held-out seeds)", { timeout: 180000 });
const evalTxt = await page.locator("text=Evaluation (held-out seeds)").locator("..").innerText();
console.log("training:", evalTxt.replace(/\n/g, " | "));
await page.screenshot({ path: "/tmp/shot-train.png", fullPage: true });

// The trained policy should now be selectable
await page.waitForTimeout(1000);
const strat = await page.locator("#strategy").inputValue();
console.log("strategy auto-switched to:", strat);
await page.getByRole("button", { name: /Run simulation/i }).click();
await page.waitForSelector("header:has-text('RL agent')", { timeout: 30000 });
const badge = await page.locator("h2:has-text('Cabin')").locator("..").innerText();
console.log("RL replay header:", badge.replace(/\n/g, " "));
await page.screenshot({ path: "/tmp/shot-rl.png", fullPage: true });

console.log("CONSOLE ERRORS:", errors.length ? errors : "none");
await browser.close();

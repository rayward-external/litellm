import { describe, it, expect } from "vitest";
import { pluginSchema } from "./schema";

const withUrl = (url: string) => pluginSchema.safeParse({ name: "n", display_name: "d", url });

describe("pluginSchema url validation", () => {
  it("accepts http and https URLs", () => {
    expect(withUrl("https://example.com/plugin").success).toBe(true);
    expect(withUrl("http://example.com").success).toBe(true);
  });

  it("accepts protocol-relative and www URLs", () => {
    expect(withUrl("//example.com/plugin").success).toBe(true);
    expect(withUrl("www.example.com").success).toBe(true);
  });

  it("rejects javascript: URLs, including the //host bypass form", () => {
    expect(withUrl("javascript:alert(1)").success).toBe(false);
    expect(withUrl("javascript://example.com/%0aalert(1)").success).toBe(false);
  });

  it("rejects other non-http(s) schemes", () => {
    expect(withUrl("data:text/html,<script>alert(1)</script>").success).toBe(false);
    expect(withUrl("vbscript:alert(1)").success).toBe(false);
    expect(withUrl("file:///etc/passwd").success).toBe(false);
  });
});

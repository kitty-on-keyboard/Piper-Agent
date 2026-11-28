import { describe, expect, it } from "vitest";
import { filterFuzzy } from "../fuzzy";

describe("fuzzy", () => {
  it("ranks basename and boundary matches above loose matches", () => {
    const files = [
      "docs/notes/something_else.md",
      "src/surface/sidecar.cpp",
      "src/app/host_helpers.cpp",
      "scripts/archive_stuff.py",
    ];
    const hits = filterFuzzy("host", files);
    expect(hits[0]).toBe("src/app/host_helpers.cpp");
  });

  it("returns empty when query characters do not appear in order", () => {
    expect(filterFuzzy("xyz", ["src/main.cpp", "README.md"])).toEqual([]);
  });
});

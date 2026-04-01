#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

/// Push constants for screen geometry.
layout(push_constant) uniform PushConstants {
    uvec2 resolution;   // text grid size in cells (e.g. 160, 50)
    uvec2 cell_pixels;  // pixel size of a single cell (e.g. 8, 16)
    uvec2 atlas_cells;  // glyph atlas grid (e.g. 256, 256)
} pc;

/// Cell data SSBO / 3 × uint per cell: [codepoint, fg_rgb, bg_rgb]
layout(std430, set = 0, binding = 0) readonly buffer CellData {
    uint cells[];
};

/// Glyph atlas / R8 texture, each glyph in a 16×16 cell.
layout(set = 0, binding = 1) uniform sampler2D glyph_atlas;

vec3 unpack_rgb(uint packed) {
    return vec3(
        float((packed >> 16u) & 0xFFu) / 255.0,
        float((packed >>  8u) & 0xFFu) / 255.0,
        float( packed         & 0xFFu) / 255.0
    );
}

void main() {
    // Total screen size in pixels.
    vec2 screen_px = vec2(pc.resolution * pc.cell_pixels);
    vec2 pixel = v_uv * screen_px;

    // Determine which cell this fragment belongs to.
    ivec2 cell = ivec2(floor(pixel / vec2(pc.cell_pixels)));

    // Out-of-bounds → black.
    if (cell.x < 0 || cell.y < 0 ||
        cell.x >= int(pc.resolution.x) ||
        cell.y >= int(pc.resolution.y))
    {
        o_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Read cell data from SSBO.
    uint idx = uint(cell.y) * pc.resolution.x + uint(cell.x);
    uint codepoint = cells[idx * 3u + 0u];
    uint fg_packed = cells[idx * 3u + 1u];
    uint bg_packed = cells[idx * 3u + 2u];

    // Position within the cell [0..1).
    vec2 cell_frac = fract(pixel / vec2(pc.cell_pixels));

    // Glyph atlas coordinates.
    //   Atlas layout: 256 columns × 256 rows of glyphs.
    //   Code point → col = cp & 0xFF, row = (cp >> 8) & 0xFF.
    uint glyph_col = codepoint & 0xFFu;
    uint glyph_row = (codepoint >> 8u) & 0xFFu;
    vec2 glyph_origin = vec2(float(glyph_col), float(glyph_row))
                      / vec2(pc.atlas_cells);
    vec2 glyph_size   = 1.0 / vec2(pc.atlas_cells);
    vec2 atlas_uv     = glyph_origin + cell_frac * glyph_size;

    float alpha = texture(glyph_atlas, atlas_uv).r;

    vec3 fg = unpack_rgb(fg_packed);
    vec3 bg = unpack_rgb(bg_packed);

    o_color = vec4(mix(bg, fg, alpha), 1.0);
}
#version 450

/// Fragment shader for text rendering.
///
/// Reads cell data from an SSBO (codepoint, foreground RGB, background RGB)
/// and glyph bitmaps from a sampled R8 texture atlas. Outputs the correct
/// foreground or background color for each pixel.
///
/// ## Inputs
///
/// * `v_uv` (location 0) - UV coordinates from the fullscreen triangle
///   vertex shader. NOT USED -- we use `gl_FragCoord.xy` instead for
///   pixel-perfect integer arithmetic.
///
/// ## Push constants
///
/// * `resolution` - Text grid size in cells [cols, rows]
/// * `cell_pixels` - Cell dimensions in pixels [8, 16]
/// * `atlas_cells` - Atlas grid dimensions [256, 256]
/// * `screen_pixels` - Window size in pixels [width, height]
///
/// ## Descriptor set (set 0)
///
/// * Binding 0: `CellData` SSBO (std430, readonly)
///   - `cells[]` - Flat array of u32 values.
///   - For cell at (col, row):
///     - `cells[(row * cols + col) * 3 + 0]` = codepoint
///     - `cells[(row * cols + col) * 3 + 1]` = foreground RGB (0x00RRGGBB)
///     - `cells[(row * cols + col) * 3 + 2]` = background RGB (0x00RRGGBB)
///
/// * Binding 1: `glyph_atlas` combined image sampler (R8_UNORM, nearest)
///   - 4096x4096 pixel atlas.
///   - Code point `cp` maps to atlas position:
///     - x = (cp & 0xFF) * 16 + intra_x
///     - y = ((cp >> 8) & 0xFF) * 16 + intra_y
///   - Pixel value > 0.5 -> foreground, else background.
///
/// ## Algorithm
///
/// ```text
/// 1. pixel = ivec2(gl_FragCoord.xy)              // exact integer pixel
/// 2. text_area = resolution * cell_pixels         // total text area in px
/// 3. if pixel outside text_area -> output black
/// 4. cell = pixel / cell_pixels                   // integer division
/// 5. idx = cell.y * resolution.x + cell.x        // flat SSBO index
/// 6. Read (codepoint, fg, bg) from SSBO
/// 7. intra = pixel % cell_pixels                  // position within cell
/// 8. atlas_texel = glyph_origin + intra           // atlas lookup coords
/// 9. alpha = texelFetch(atlas, atlas_texel, 0).r  // exact texel, no filter
/// 10. output = mix(bg, fg, alpha)                 // blend fg/bg
/// ```

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(push_constant) uniform PushConstants {
    uvec2 resolution;     // text grid in cells (e.g. 160, 50)
    uvec2 cell_pixels;    // screen cell size (8, 16)
    uvec2 atlas_cells;    // atlas grid (256, 256)
    uvec2 screen_pixels;  // window size in pixels
} pc;

/// Cell data SSBO. Flat u32 array, 3 values per cell:
/// [codepoint, fg_packed_rgb, bg_packed_rgb]
layout(std430, set = 0, binding = 0) readonly buffer CellData {
    uint cells[];
};

/// Glyph atlas: 4096x4096 R8_UNORM texture.
/// Sampled with NEAREST filter for pixel-perfect glyph rendering.
layout(set = 0, binding = 1) uniform sampler2D glyph_atlas;

/// Unpack a 24-bit RGB value (0x00RRGGBB) into a vec3 in [0, 1] range.
vec3 unpack_rgb(uint packed) {
    return vec3(
        float((packed >> 16u) & 0xFFu) / 255.0,
        float((packed >>  8u) & 0xFFu) / 255.0,
        float( packed         & 0xFFu) / 255.0
    );
}

void main() {
    // Use gl_FragCoord.xy for exact integer pixel coordinates.
    // This avoids UV interpolation precision issues.
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    // Compute the total text area bounds in pixels.
    ivec2 text_area = ivec2(pc.resolution * pc.cell_pixels);
    if (pixel.x >= text_area.x || pixel.y >= text_area.y) {
        // Outside the text grid -> solid black border.
        o_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Determine which cell this pixel belongs to (integer division).
    ivec2 cp = ivec2(pc.cell_pixels);
    ivec2 cell = pixel / cp;

    // Read cell data from the SSBO.
    uint idx = uint(cell.y) * pc.resolution.x + uint(cell.x);
    uint codepoint = cells[idx * 3u + 0u];
    uint fg_packed = cells[idx * 3u + 1u];
    uint bg_packed = cells[idx * 3u + 2u];

    // Compute the position within the cell (integer modulo).
    ivec2 intra = pixel % cp;

    // Look up the glyph in the atlas.
    // Each glyph cell is 16x16 pixels in the atlas.
    uint glyph_col = codepoint & 0xFFu;
    uint glyph_row = (codepoint >> 8u) & 0xFFu;
    ivec2 atlas_texel = ivec2(
        int(glyph_col) * 16 + intra.x,
        int(glyph_row) * 16 + intra.y
    );

    // texelFetch: exact texel lookup, zero filtering, zero precision loss.
    float alpha = texelFetch(glyph_atlas, atlas_texel, 0).r;

    // Blend foreground and background based on the atlas alpha.
    vec3 fg = unpack_rgb(fg_packed);
    vec3 bg = unpack_rgb(bg_packed);
    o_color = vec4(mix(bg, fg, alpha), 1.0);
}
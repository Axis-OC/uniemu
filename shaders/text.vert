#version 450

/// Vertex shader for fullscreen triangle rendering.
///
/// Generates a single triangle that covers the entire screen using
/// only `gl_VertexIndex` (no vertex buffer needed).
///
/// ## Vertex positions (clip space)
///
/// ```text
/// gl_VertexIndex = 0: position = (-1, -1), uv = (0, 0)  -- bottom-left
/// gl_VertexIndex = 1: position = ( 3, -1), uv = (2, 0)  -- far right
/// gl_VertexIndex = 2: position = (-1,  3), uv = (0, 2)  -- far top
/// ```
///
/// The triangle extends beyond the screen bounds, but the rasterizer
/// clips it to the viewport. The visible area is exactly [-1,1] x [-1,1],
/// which corresponds to UVs [0,1] x [0,1].
///
/// ## Why a single triangle instead of a quad?
///
/// A single triangle avoids the diagonal seam artifact that can occur
/// with two-triangle quads due to floating-point rasterisation
/// differences along the shared edge. It also eliminates one vertex
/// fetch and one triangle setup.
///
/// ## Output
///
/// * `v_uv` (location 0) - UV coordinates in [0, 1] range.
///   Not used by text.frag (which uses gl_FragCoord instead) but
///   kept for potential future use.

layout(location = 0) out vec2 v_uv;

void main() {
    // Compute UV from vertex index using bit manipulation:
    // Index 0: uv = (0, 0)
    // Index 1: uv = (2, 0)
    // Index 2: uv = (0, 2)
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

    // Convert UV [0, 2] to clip space [-1, 3]:
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
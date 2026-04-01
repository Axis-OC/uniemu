#version 450

/// Fullscreen triangle / no vertex buffer needed.
/// Three vertices with IDs 0,1,2 cover the entire screen.

layout(location = 0) out vec2 v_uv;

void main() {
    // Generate fullscreen triangle UVs from vertex index.
    //   ID=0 → (0,0)  ID=1 → (2,0)  ID=2 → (0,2)
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y for Vulkan's top-left origin.
    v_uv.y = 1.0 - v_uv.y;
}
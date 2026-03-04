#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 2) in vec4 in_center_radius; // xyz center, w radius
layout(location = 3) in float in_speed;

uniform mat4 mvp;
uniform mat3 normal_matrix;

out vec3 v_norm;
out float v_speed;

void main() {
    vec3 world_pos = in_center_radius.xyz + in_pos * in_center_radius.w;
    gl_Position = mvp * vec4(world_pos, 1.0);
    v_norm = normalize(normal_matrix * in_norm);
    v_speed = in_speed;
}

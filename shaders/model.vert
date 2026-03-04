#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;

uniform mat4 mvp;
uniform mat3 normal_matrix;

out vec3 v_norm;
out vec3 v_pos;

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_norm = normalize(normal_matrix * in_norm);
    v_pos = in_pos;
}

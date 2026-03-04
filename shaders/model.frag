#version 330 core
in vec3 v_norm;
in vec3 v_pos;

uniform vec3 light_dir;

out vec4 fragColor;

void main() {
    vec3 n = normalize(v_norm);
    float diff = max(dot(n, normalize(light_dir)), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 axis_tint = abs(n);
    vec3 base = vec3(0.58, 0.62, 0.70) + axis_tint * 0.24 + vec3(0.05, 0.07, 0.10) * hemi;
    float stripe = 0.5 + 0.5 * sin(v_pos.y * 5.0);
    vec3 color = base * (0.30 + 0.70 * diff) + vec3(0.08, 0.09, 0.12) * stripe * 0.28;
    fragColor = vec4(color, 1.0);
}

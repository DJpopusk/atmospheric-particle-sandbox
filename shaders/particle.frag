#version 330 core
in vec3 v_norm;
in float v_speed;

uniform vec3 light_dir;

out vec4 fragColor;

void main() {
    float diff = max(dot(normalize(v_norm), normalize(light_dir)), 0.0);
    float t = clamp(v_speed * 0.55, 0.0, 1.0);
    vec3 slow = vec3(0.30, 0.86, 1.00);
    vec3 fast = vec3(1.00, 0.72, 0.22);
    vec3 base = mix(slow, fast, t);
    vec3 color = base * (0.55 + 0.45 * diff);
    fragColor = vec4(color, 1.0);
}

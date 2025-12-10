#version 430

in vec3 fragColor;
in vec3 FragPos;
in vec3 vNormal;

out vec4 outColor;

uniform vec3 lightDir;   // dirección del sol
uniform vec3 camPos;
uniform vec3 lightColor;

void main()
{
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(vNormal);
    vec3 L = normalize(-lightDir); // rayos solares
    float diff = max(dot(norm, L), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(camPos - FragPos);
    vec3 reflectDir = reflect(-L, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * fragColor;
    outColor = vec4(result, 1.0);
}
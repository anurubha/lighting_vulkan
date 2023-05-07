#version 450 

layout (location = 0) in vec3 gPosition; //in world space
layout (location = 1) in vec3 gNormal;
layout (location = 2) in vec2 gTexCoord; 
layout (location = 3) in vec4 gtangent;

layout( set = 0, binding = 0 ) uniform UScene 
	{ 
		mat4 camera; 
		mat4 projection; 
		mat4 projCam; 


		vec3 cameraPosition;
		vec3 lightPosition;
		vec3 lightColor;
		vec3 ambientColor;
	} uScene; 


layout( set = 1, binding = 0 ) uniform sampler2D BaseColorSampler;
layout( set = 1, binding = 1 ) uniform sampler2D RoughnessSampler; 
layout( set = 1, binding = 2 ) uniform sampler2D MetalnessSampler; 
layout( set = 1, binding = 3 ) uniform sampler2D NormalMapSampler;

layout( location = 0 ) out vec4 oColor; 

void main() 
{ 
	// reading and converting from [0, 1] to [-1, 1]
	vec3 mapNormal = normalize(2 * texture( NormalMapSampler, gTexCoord ).rgb - 1.0);
	vec3 vNormal = normalize(gNormal);
	vec4 tangent = normalize(gtangent);
	vec3 bitangent = normalize(cross(vNormal, tangent.xyz) * tangent.w);

	vec3 normal = normalize( mat3( tangent.xyz, bitangent, vNormal) * mapNormal); 
	
	vec3 lightDirection = normalize(uScene.lightPosition - gPosition); 
	vec3 viewDirection = normalize(uScene.cameraPosition - gPosition);	
	vec3 halfVector = normalize(viewDirection + lightDirection);
	//normal = normalize(gNormal);

	vec3 basecolor = texture( BaseColorSampler, gTexCoord ).rgb;
	highp float roughness = texture( RoughnessSampler, gTexCoord ).r;
	highp float metalness = texture( MetalnessSampler, gTexCoord ).r;	
	highp float shininess =  max(2/(pow(roughness,4)), 0.0001) - 2;


	// Dot Products
	float NoH = max(dot(normal, halfVector), 0.0);
	float NoV = max(dot(normal, viewDirection), 0.0);
	float NoL = max(dot(normal, lightDirection), 0.0);
	float VoH = dot(viewDirection, halfVector);

	// Ambient Light
	vec3 AmbientLight = uScene.ambientColor * basecolor;

	// Fresnel term - F, evaluate using the Schlick approximation
	vec3 F0 = (1-metalness) * vec3(0.04f, 0.04f, 0.04f) + metalness * basecolor;
	vec3 F = F0 + (1 - F0) * pow((1 - VoH), 5);

	// Lambertian Diffuse
	vec3 LDiffuse = (basecolor/3.14159265) * (vec3(1.0f,1.0f,1.0f) - F ) * (1.0f - metalness);

	// Normal Distribution Function - D
	float D = ((shininess + 2)/(2.0 * 3.14159265)) * (pow(NoH,shininess));

	// Masking term from the Cook-Torrance model - G
	float G = min (1, min((2 * NoH * NoV / VoH) , (2 * NoH * NoL / VoH)));

	// microfacet BRDF model
	vec3 Fr = LDiffuse + (D * F * G )/ max((4 * NoV * NoL), 0.0001); 

	oColor = vec4(AmbientLight + Fr * uScene.lightColor * NoL , 1.0f);
	//oColor = vec4(vNormal,  1.0f);

} 


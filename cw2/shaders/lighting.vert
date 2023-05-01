#version 450 

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

layout( set = 0, binding = 0 ) uniform UScene 
	{ 
		mat4 camera; 
		mat4 projection; 
		mat4 projCam; 

		vec3 cameraPosition;
		vec3 lightPosition;
		vec4 lightColor;
		vec4 ambientColor;
	} uScene; 

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal; 
layout (location = 2) out vec2 gTexCoord; 

void main() 
{ 
	gPosition = position;
	gNormal = normalize(normal);
	gTexCoord = texcoord;

	gl_Position = uScene.projCam * vec4( position, 1.f ); 
} 
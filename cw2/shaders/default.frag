#version 450 

layout (location = 0) in vec3 gPosition;
layout (location = 1) in vec3 gNormal;

layout( set = 0, binding = 0 ) uniform UScene 
	{ 
		mat4 camera; 
		mat4 projection; 
		mat4 projCam; 

		vec3 cameraPosition;
		vec3 lightPosition;
		vec4 lightColor;
	} uScene; 


layout( location = 0 ) out vec4 oColor; 

void main() 
{ 
	// Normal as color
	oColor = vec4(normalize(gNormal),1.f);

	// Light Direction as color
	//oColor = vec4( normalize(uScene.lightPosition - gPosition), 1.f ); 

	// Camera Direction as color
	//oColor = vec4( normalize(uScene.cameraPosition - gPosition), 1.f );

} 

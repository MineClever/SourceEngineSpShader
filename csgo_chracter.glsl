//this shader is used for source engine games
//writed by MineClever aka 悍匪 Bandit
//if you find bugs , you can tell in Steam : https://steamcommunity.com/id/chinsesbandit/


/*
Mask1 could control everything nice
Mask2 just used as fixed reflection term.
*/

import lib-utils.glsl
import lib-env.glsl
import lib-normal.glsl
import lib-alpha.glsl
import lib-vectors.glsl
import lib-sparse.glsl
import lib-sampler.glsl

#undef INV_U

// fix look different from hlmv
	// so buggy!!
#undef LINEAR_LOOK
#ifndef LINEAR_LOOK
	#define INV_GAMMA
#endif

// fix look different from hlmv
#define BOOST_BASE
#ifdef BOOST_BASE
	#define BOOST_NUM 10
#endif
#ifndef BOOST_BASE
	#define BOOST_NUM 1
#endif

#define cOverbright 2.0f
#define cOOOverbright 0.5f
#define g_vModelOrigin vec3(0.0,0.0,0.0) //model origin

/*
//--------Switch Parameters ---------------------------------------------------//
*/

//: param auto camera_view_matrix_it
uniform mat4 uniform_camera_view_matrix_it;

//: param auto camera_view_matrix
uniform mat4 uniform_camera_view_matrix;

//: state cull_face off

//: param custom { "default": false, "label": "Alpha Test(Not Work In Game)", "group": "Switch Parameters" }
uniform bool bHasAlpha;

//: param custom { "default": true, "label": "Use Mask_1 map", "group": "Switch Parameters" }
uniform bool MASKS1;

//: param custom { "default": false, "label": "Use Mask_2 map", "group": "Switch Parameters" }
uniform bool MASKS2;

//: param custom { "default": false, "label": "Use IBL Sun", "group": "Switch Parameters" }
uniform bool bIBL;

//: param custom { "default": false, "label": "Use IBL Sun Color", "group": "Switch Parameters" }
uniform bool bIBL_Color;

//: param custom { "default": true, "label": "Use IBL Ambient (Game)", "group": "Switch Parameters" }
uniform bool bAmbientLight;

//: param custom { "default": true, "label": "Use Ambient Color (hlmv)", "group": "Switch Parameters" }
uniform bool bAmbient;

//: param custom { "default": [0.2,0.2,0.2], "label": "Ambient Color", "widget": "color", "group": "Switch Parameters" }
uniform vec3 ambient_color;


/*
//--------warptex Parameters ---------------------------------------------------//
*/

//: param custom { "default": 0.1, "label": "$WarpIndex", "min": 0.0, "max": 1.0, "group": "Warp Parameters" }
uniform float g_fWarpIndex;

//: param custom { "default": false, "label": "Fresnel Warp", "group": "Warp Parameters" }
uniform bool FRESNELRANGESTEXTURE;

//: param custom { "default": "fresnelranges", "label": "FresnelWarp Map", "usage": "texture", "group": "Warp Parameters" }
uniform sampler2D FresnelRangesSampler;

//: param custom { "default": false, "label": "Light Warp(Not work in Game)", "group": "Warp Parameters" }
uniform bool bDoLightingWarp;

//: param custom { "default": "fresnelranges", "label": "LightWarp Tex", "usage": "texture", "group": "Warp Parameters" }
uniform sampler2D LightingWarpSampler;

//: param custom { "default": false, "label": "Phong Warp", "group": "Warp Parameters" }
uniform bool PHONGWARPTEXTURE;

//: param custom { "default": "fresnelranges", "label": "PhongWarp Tex", "usage": "texture", "group": "Warp Parameters" }
uniform sampler2D PhongWarpSampler;

//: param custom { "default": [0.35, 0.75, 1.0], "label": "Fresnel Ranges", "group": "Warp Parameters" }
uniform vec3 g_vFresnelRanges;

//-------- Diffuse Terms Parameters ---------------------------------------------------//


//: param custom { "default": false, "label": "Use HalfLambert(Not work in Game)", "group": "Diffuse Terms Parameters" }
uniform bool HALFLAMBERT;

//: param custom { "default": 1.0, "label": "$Metalness", "min": 0.0, "max": 1.0, "group": "Diffuse Terms Parameters" }
uniform float g_fMetalness;

//: param custom { "default": [1.0,1.0,1.0], "label": "$color", "widget": "color", "group": "Diffuse Terms Parameters" }
uniform vec3 r_DiffuseTint;

#define g_DiffuseTint r_DiffuseTint

//-------- Unique Phong Parameters ---------------------------------------------------//

//: param custom { "default": true, "label": "Use Phong Specular", "group": "Phong Parameters" }
uniform bool PHONG;

//: param custom { "default": false, "label": "Opacity as PhongMask", "group": "Phong Parameters" }
uniform bool BASEALPHAPHONGMASK;

//: param custom { "default": 1.0, "label": "$PhongBoost", "min": 0.0, "max": 255.0, "group": "Phong Parameters" }
uniform float f_fPhongBoost;
#define g_fPhongBoost f_fPhongBoost*BOOST_NUM

//: param custom { "default": 1.0, "label": "$PhongAlbedoBoost", "min": 0.0, "max": 255.0, "group": "Phong Parameters" }
uniform float f_fPhongAlbedoBoost;
#define g_fPhongAlbedoBoost f_fPhongAlbedoBoost*BOOST_NUM

//: param custom { "default": 5.0, "label": "$PhongExponent(Override by Mask)", "min": 1.0, "max": 150.0, "group": "Phong Parameters" }
uniform float g_fPhongExponent;

//: param custom { "default": true, "label": "Use Anisotropy", "group": "Phong Parameters" }
uniform bool ANISOTROPY;

//: param custom { "default": 0.0, "label": "$AnisotropyAmount", "min": 0.0, "max": 1.0, "group": "Phong Parameters" }
uniform float g_fAnisotropyAmount;


//-------- Rim Lighting Parameters ---------------------------------------------------//


//: param custom { "default": true, "label": "Use RimLight", "group": "RimLight Parameters" }
uniform bool RIMLIGHT;

//: param custom { "default": 1.0, "label": "$RimlightAlbedo", "min": 0.0, "max": 255.0, "group": "RimLight Parameters" }
uniform float g_fRimLightAlbedo;

//: param custom { "default": 5.0, "label": "$RimLightExponent", "min": 0.0, "max": 255.0, "group": "RimLight Parameters" }
uniform float r_fRimLightExponent;
#define g_fRimLightExponent r_fRimLightExponent

//: param custom { "default": 1.0, "label": "$RimLightBoost", "min": 0.0, "max": 255.0, "group": "RimLight Parameters" }
uniform float f_fRimLightBoost;
#define g_fRimLightBoost f_fRimLightBoost*BOOST_NUM

//: param custom { "default": 1, "label": "$RimLightTint", "widget": "color", "group": "RimLight Parameters" }
uniform vec3 r_cRimLightTint;

#define g_cRimLightTint r_cRimLightTint


//-------- Fake RimLight Parameters ---------------------------------------------------//


//: param custom { "default": false, "label": "Use Fake Rim", "group": "Fake RimLight Parameters"}
uniform bool FAKERIM;

//: param custom { "default": 1.0, "label": "$FakeRimBoost", "min": 0.0, "max": 255.0, "group": "Fake RimLight Parameters"}
uniform float f_fRimboost;
#define g_fRimboost f_fRimboost*BOOST_NUM

//: param custom { "default": [1.0,1.0,1.0], "label": "$FakeRimTint", "widget": "color", "group": "Fake RimLight Parameters"}
uniform vec3 r_cFakeRimTint;

#define g_cFakeRimTint r_cFakeRimTint

//-------- Rim Halo Parameters ---------------------------------------------------//


//: param custom { "default": [0.4, 0.5, 0.5, 0.6], "label": "$RimHaloBounds", "min": -1.0, "max": 1.0, "group": "RimHalo Parameters" }
uniform vec4 g_vRimHaloBounds;

//: param custom { "default": 1.0, "label": "$RimHaloBoost", "min": 0.0, "max": 255.0, "group": "RimHalo Parameters" }
uniform float f_fRimHaloBoost;
#define g_fRimHaloBoost f_fRimHaloBoost*BOOST_NUM



//-------- Ambient Reflection Parameters ---------------------------------------------------//


//: param custom { "default": false, "label": "Use Ambient Reflection (*fresnelMask.B)", "group": "Ambient Reflection Parameters" }
uniform bool AMBIENTREFLECTION;

//: param custom { "default": false, "label": "Use Bounce Color", "group": "Ambient Reflection Parameters" }
uniform bool USEBOUNCECOLOR;

//: param custom { "default": 1.0, "label": "$AmbientReflectionBoost", "min": 0.0, "max": 255.0, "group": "Ambient Reflection Parameters" }
uniform float f_fAmbientBounceBoost;
#define r_fAmbientBounceBoost f_fAmbientBounceBoost*BOOST_NUM

//: param custom { "default": [1.0,1.0,1.0], "label": "$AmbientReflectionBounceColor", "widget": "color", "group": "Ambient Reflection Parameters" }
uniform vec3 r_cBounce;

#define g_cBounce r_cBounce


//: param custom { "default": [0.0, 42.0, 0.0], "label": "$AmbientReflectionBounceCenter", "min": -255.0, "max": 255.0, "group": "Ambient Reflection Parameters" }
uniform vec3 g_vBounceCenter;

//-------- Hard Code Parameters ---------------------------------------------------//
#define g_fRetroReflectivityBoost 5.0f
#define g_fRetroReflectivityPower 4.0f

#define g_vShadowRimMin   0.01f //origin : 0.01f
#define g_vShadowRimMax   0.05f //origin : 0.05f
// #define g_vViewFakeRimDir vec3( 0.7f, 1.0f, 0.0f ) //engine value
#define g_vViewFakeRimDir vec3( 0.0f, 1.0f, 0.7f ) // Substance Painter Z should be front face as Source Engine X ?


//-------- Shadow Parameters Parameters ---------------------------------------------------//

//: param custom { "default": false, "label": "Use ShadowSaturation", "group": "Shadow Parameters Parameters" }
uniform bool SHADOWSATURATION;

//: param custom { "default": 1, "label": "$ShadowTint", "widget": "color", "group": "Shadow Parameters Parameters" }
uniform vec4 r_cShadowTint;

#define g_cShadowTint r_cShadowTint

//: param custom { "default": 0.0, "label": "$shadowcontrast", "min": 0.0, "max": 1.0, "group": "Shadow Parameters Parameters" }
uniform float g_fShadowScale;

//: param custom { "default": 1.0, "label": "$ShadowSaturation", "min": -1.0, "max": 1.0, "group": "Shadow Parameters Parameters" }
uniform float g_fShadowSaturation;

//: param custom { "default": 1.0, "label": "$ShadowRimBoost", "min": 0.0, "max": 255.0, "group": "Shadow Parameters Parameters" }
uniform float f_fShadowRimBoost;
#define g_fShadowRimBoost f_fShadowRimBoost*BOOST_NUM

//: param custom { "default": [0.4, 0.5, 0.5, 0.6], "label": "$ShadowSaturationBounds", "min": -1.0, "max": 1.0, "group": "Shadow Parameters Parameters" }
uniform vec4 g_vShadowSaturationBounds;

//-------- Unique Envmap Parameters ---------------------------------------------------//
//: param custom { "default": false, "label": "Use EnvMap", "group": "Envmap Parameters" }
uniform bool ENVMAP;

//: param custom { "default": false, "label": "Opacity as EnvMask", "group": "Envmap Parameters"}
uniform bool BASEALPHAENVMASK;

//: param custom { "default": 1.0, "label": "$EnvmapLightScale", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_fEnvmapLightScale;

//: param custom { "default": 0.0, "label": "$EnvmapLightScaleMin", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_vEnvmapLightScaleMin;

//: param custom { "default": 1.0, "label": "$EnvmapLightScaleMax", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_vEnvmapLightScaleMax;

//: param custom { "default": 0.0, "label": "$EnvmapContrast", "min": -10.0, "max": 10.0, "group": "Envmap Parameters" }
uniform float g_fEnvmapContrast;

//: param custom { "default": 0.0, "label": "$EnvmapSaturation", "min": -1.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_fEnvmapSaturation;

//: param custom { "default": 1, "label": "$EnvmapTint", "widget": "color", "group": "Envmap Parameters" }
uniform vec3 r_cEnvmapTint;

#define g_cEnvmapTint r_cEnvmapTint

//-------- SelfIllum Parameters ---------------------------------------------------//

//: param custom { "default": false, "label": "Enable SelfIllum", "group": "SelfIllum Parameters" }
uniform bool SELFILLUM;

//: param custom { "default": false, "label": "Opacity as SelfIllumMask", "group": "SelfIllum Parameters" }
uniform bool BASEALPHASELFILLUMMASK;

//: param custom { "default": 1.0, "label": "$SelfIllumBoost", "min": 0.0, "max": 255.0, "group": "SelfIllum Parameters" }
uniform float f_fSelfIllumBoost;
#define g_fSelfIllumBoost f_fSelfIllumBoost*BOOST_NUM


/*
//-------- Lights ----------------------------------------------------//
*/

//: param custom { "default": 1, "label": "Light Num", "min": 1, "max": 4, "group": "Lights Parameters" }
uniform int NUM_LIGHTS;

//: param custom { "default": 1.0, "label": "Light Lum Boost", "min": 1.0, "max": 1024.0, "group": "Lights Parameters" }
uniform float r_lightBoost;

//: param custom { "default": 2, "label": "Light Falloff Type.1=None,2=Distance,3=Expoent", "min": 1, "max": 3, "group": "Lights Parameters" }
uniform int r_DistanceType;

//: param custom { "default": [0.0, 1.0, 1.0], "label": "Sun Position", "min": -1.0, "max": 1.0, "group": "Lights Parameters" }
uniform vec3 Sunlight_0;

//: param custom { "default": [1.0,1.0,1.0], "label": "Sun Color", "widget": "color", "group": "Lights Parameters" }
uniform vec3 SunlightColor_0;

/*
//-------- Other lights ----------------------------------------------------//
*/
//: param custom { "default": [0.0, 10.0, 10.0], "label": "Light Position 1", "min": -10.0, "max": 10.0, "group": "Extra Lights1" }
uniform vec3 lightPosition_1;

//: param custom { "default": [1.0,1.0,1.0], "label": "Light Color 1", "widget": "color", "group": "Extra Lights1" }
uniform vec3 lightColor_1;

//: param custom { "default": [0.0, 0.0, -10.0], "label": "Light Position 2", "min": -10.0, "max": 10.0, "group": "Extra Lights2" }
uniform vec3 lightPosition_2;

//: param custom { "default": [0.0,0.0,0.0], "label": "Light Color 2", "widget": "color", "group": "Extra Lights2" }
uniform vec3 lightColor_2;

//: param custom { "default": [0.0, -10.0, 10.0], "label": "Light Position 3", "min": -10.0, "max": 10.0, "group": "Extra Lights3" }
uniform vec3 lightPosition_3;

//: param custom { "default": [0.0,0.0,0.0], "label": "Light Color 3", "widget": "color", "group": "Extra Lights3" }
uniform vec3 lightColor_3;

//: param auto main_light
uniform vec4 light_main;//light_0


/*
//--------texure Sampler ---------------------------------------------------//
*/

//base texture

//: param auto channel_basecolor
uniform SamplerSparse BaseTextureSampler;

//: param auto channel_opacity
uniform SamplerSparse OpacityAlphaSampler;

//: param auto channel_specular
uniform SamplerSparse SpecularSampler;

//: param auto channel_emissive
uniform SamplerSparse emissive_tex;

//: param auto channel_user0
uniform SamplerSparse EnvMapSampler;


//--------mask1 Sampler ---------------------------------------------------//
/*
Red channel - $rimlight mask
Green channel - Phong Albedo mask
Blue channel - $metalness mask
Alpha channel - $warpindex mask
*/

//: param auto channel_user2
uniform SamplerSparse RimlightSampler;

//: param auto channel_user1
uniform SamplerSparse AlbedoTintSampler;

//: param auto channel_metallic
uniform SamplerSparse MetallicSampler;

//: param auto channel_user3
uniform SamplerSparse WarpIndexMaskSampler;

//--------mask2 Sampler ---------------------------------------------------//
/*
Red channel - Shadow saturation mask
Green channel - Angle of Anisotropy
Blue channel - $envmap light scale
Alpha channel - Retroreflectivity == pow(NdotV,4.0)
*/

//: param auto channel_user4
uniform SamplerSparse ShadowSaturationSampler;

//: param auto channel_anisotropyangle
uniform SamplerSparse AnisotropyAngleSampler;

//: param auto channel_user5
uniform SamplerSparse LightScaleSampler;

//: param auto channel_user6
uniform SamplerSparse RetroreflectivitySampler;

//----------------helpe to fix----------------
//: param auto channel_user7
uniform SamplerSparse AmbientReflectionBoostSampler;


//hlsl function

float saturate(float v)
{
  return clamp(v, 0.0, 1.0);
}

vec3 saturate(vec3 v)
{
  return clamp(v, 0.0, 1.0);
}

void sincos(float x, out float s, out float c)
{
 s = sin (x);
 c = cos (x);
}

float rsqrt(float a)
{
  // return pow(a, -0.5);
  return inversesqrt(a);
}

//custom function

// 2.2 gamma conversion routines
float LinearToGamma( float f1linear )
{
	return pow( f1linear, 1.0f / 2.2f );
}

vec3 LinearToGamma( vec3 f3linear )
{
	return pow( f3linear, vec3(1.0f / 2.2f) );
}

vec4 LinearToGamma( vec4 f4linear )
{
	return vec4( pow( f4linear.xyz, vec3(1.0f / 2.2f) ), f4linear.w );
}

float GammaToLinear( float gamma )
{
	return pow( gamma, 2.2f );
}

vec3 GammaToLinear( vec3 gamma )
{
	return (pow( gamma, vec3(2.2f) ));
}

vec4 GammaToLinear( vec4 gamma )
{
	return vec4( pow( gamma.xyz, vec3(2.2f) ), gamma.w );
}

//source engine fresnel Ranges

vec3 CalcReflectionVectorUnnormalized( vec3 normal, vec3 eyeVector )
{
	// FIXME: might be better of normalizing with a normalizing cube map and
	// get rid of the dot( normal, normal )
	// compute reflection vector r = 2 * ((n dot v)/(n dot n)) n - v
	//  multiply all values through by N.N.  uniformly scaling reflection vector won't affect result
	//  since it is used in a cubemap lookup
	return (2.0*(dot( normal, eyeVector ))*normal) - (dot( normal, normal )*eyeVector);
}


float Fresnel2( vec3 vNormal, vec3 vEyeDir )
{
	float fresnel = 1-saturate( dot( vNormal, vEyeDir ) );				// 1-(N.V) for Fresnel term
	return fresnel * fresnel;											// Square for a more subtle look
}

float Fresnel(vec3 vNormal, vec3 vEyeDir, vec3 vRanges)
{

	float result, f = Fresnel2( vNormal, vEyeDir );			// Traditional Fresnel

	if ( f > 0.5f )
		result = mix( vRanges.y, vRanges.z, (2*f)-1 );		// Blend between mid and high values
	else
        result = mix( vRanges.x, vRanges.y, 2*f );			// Blend between low and mid values

	return result;
}

float Fresnel4( vec3 vNormal, vec3 vEyeDir )
{
	float fresnel = 1-saturate( dot( vNormal, vEyeDir ) );				// 1-(N.V) for Fresnel term
	fresnel = fresnel * fresnel;										// Square
	return fresnel * fresnel;											// Square again for a more subtle look
}

vec3 saturateColor( vec3 c1, float fsat )
{
	vec3 finalColor = c1;

	if ( fsat != 0 )
	{

		// perceptual luminance
		vec3 lum = vec3( 0.299, 0.587, 0.114 );

		vec3 c2 = pow( c1,vec3(4));
		float luminance1 = dot( c1, lum );

		if ( fsat < 0 )
		{
			finalColor = mix( c1, vec3(luminance1), vec3(-fsat) );
		}
		else
		{
			float luminance2 = dot( c2, lum );
			luminance2 = max( luminance2, 0.000001f );
			c2 = c2 * luminance1 / luminance2;
			finalColor = mix( c1, c2, fsat );
		}
	}
	return finalColor;
}

vec3 tintColor( vec3 c1, vec3 tint, float amt )
{
   // perceptual luminance
   vec3 lum = vec3( 0.299, 0.587, 0.114 );

   vec3 c2 = tint;
   float luminance1 = dot( c1, lum );
   float luminance2 = dot( c2, lum );
   luminance2 = max( luminance2, 0.000001f );
   c2 = c2 * luminance1 / luminance2 ;
   return mix( c1, c2, amt );
}

vec3 desaturateColor( vec3 c1 )
{
   // perceptual luminance
   vec3 lum = vec3( 0.299, 0.587, 0.114 );

   return vec3(dot( c1, lum ));
}

float sampleWithDefault(SamplerSparse sampler, SparseCoord coord, float defaultValue)
{
	vec2 value = textureSparse(sampler, coord).rg;
	return value.r + defaultValue * (1.0 - value.g);
}

//lighting program

struct PixelShaderLightInfo
{
	vec4 color;
	vec4 pos;
    vec4 atten;//x=1,y=d,z=d*d
};

PixelShaderLightInfo g_cLightInfo[3];


void PropagateLights(vec3 worldPos, float r_lightBoost) //there are all light uniform parameter ,it is easy to load all set
{
    // First local light will always be forced to a directional light in CS:GO

	if (bIBL)
	{
		g_cLightInfo[0].pos.xyz 	= (light_main.xyz  + worldPos ); //scaled
		if (!bIBL_Color)
			g_cLightInfo[0].color.rgb   = (SunlightColor_0 * r_lightBoost);
		else
			g_cLightInfo[0].color.rgb	= (envIrradiance(light_main.xyz) * r_lightBoost);

	} else
	{
		g_cLightInfo[0].pos.xyz     = (normalize(Sunlight_0)  + worldPos);
		g_cLightInfo[0].color.rgb   = (SunlightColor_0 * r_lightBoost);
	}

	g_cLightInfo[1].color.rgb   = (lightColor_1 * r_lightBoost);
	g_cLightInfo[1].pos.xyz     = (lightPosition_1);
	g_cLightInfo[2].color.rgb   = (lightColor_2 * r_lightBoost);
	g_cLightInfo[2].pos.xyz     = (lightPosition_2);

	vec3 temp_color = (lightColor_3 * r_lightBoost);
	vec3 temp_pos = (lightPosition_3);
    //pos_3
	g_cLightInfo[1].pos.w       = temp_pos.x;
	g_cLightInfo[2].color.w     = temp_pos.y;
    g_cLightInfo[2].pos.w       = temp_pos.z;
    //color_3
	g_cLightInfo[0].color.w     = temp_color.r;
    g_cLightInfo[0].pos.w       = temp_color.g;
    g_cLightInfo[1].color.w     = temp_color.b;
	g_cLightInfo[0].atten = vec4(1.0,0.0,0.0,1.0);
	g_cLightInfo[1].atten = vec4(1.0,0.0,0.0,0.0);
	g_cLightInfo[2].atten = vec4(1.0,0.0,0.0,0.0);
	if (r_DistanceType == 1)
	{
		g_cLightInfo[0].atten = vec4(1.0,0.0,0.0,1.0);
		g_cLightInfo[1].atten = vec4(1.0,0.0,0.0,0.0);
		g_cLightInfo[2].atten = vec4(1.0,0.0,0.0,0.0);
	}
	if (r_DistanceType == 2)
	{
		g_cLightInfo[0].atten = vec4(0.0,1.0,0.0,0.0);
		g_cLightInfo[1].atten = vec4(0.0,1.0,0.0,1.0);
		g_cLightInfo[2].atten = vec4(0.0,1.0,0.0,0.0);
	}
	if (r_DistanceType == 3)
	{
		g_cLightInfo[0].atten = vec4(0.0,0.0,1.0,0.0);
		g_cLightInfo[1].atten = vec4(0.0,0.0,1.0,0.0);
		g_cLightInfo[2].atten = vec4(0.0,0.0,1.0,1.0);
	}
}

float VertexAttenInternal( vec3 worldPos, int lightNum ) //force set light as a point for substance
{
	float result = 0.0f;
	vec3 lightDir;
	// Get light direction
	if ( lightNum == 3 )
	{
		// Unpack light 3 from w components...
		vec3 vLight3Pos = vec3( g_cLightInfo[1].pos.w, g_cLightInfo[2].color.w, g_cLightInfo[2].pos.w );
		lightDir = ( vLight3Pos - worldPos );
	} else
		lightDir = g_cLightInfo[lightNum].pos.xyz - worldPos;

	// Get light distance squared.
	float lightDistSquared = dot( lightDir, lightDir );

	// Get 1/lightDistance
	float ooLightDist = rsqrt( lightDistSquared );

	// Normalize light direction
	lightDir *= ooLightDist;

	vec3 vDist;
    //use xbox360 way
    vDist.x = 1;
    vDist.y = lightDistSquared * ooLightDist; //d
    vDist.z = lightDistSquared; //d*d
    //flDist.w = ooLightDist; // 1/d

	float flDistanceAtten;

    if (lightNum == 3)
        flDistanceAtten = 1.0f / dot( vec3(g_cLightInfo[0].atten.w,g_cLightInfo[1].atten.w,g_cLightInfo[2].atten.w), vDist );
	else flDistanceAtten = 1.0f / dot( g_cLightInfo[lightNum].atten.xyz, vDist );

	return flDistanceAtten;
}

float GetVertexAttenForLight( vec3 worldPos, int lightNum)
{
	float result = 0.0f;
    result = VertexAttenInternal( worldPos, lightNum );
	return result;
}


//for csgo
float SoftenCosineTerm( float flDot )
{
	return ( flDot + ( flDot * flDot ) ) * 0.5;
	//return rsqrt( flDot ) * ( flDot * flDot );

	//#define SOFTEN_COSINE_EXP 0.5
	//return pow( flDot, SOFTEN_COSINE_EXP );
}

vec3 AmbientLight( vec3 worldNormal )
{
	return envIrradiance(worldNormal);//fix function using Substance Engine
}


vec3 DiffuseTerm(bool bHalfLambert, vec3 worldNormal, vec3 lightDir,
				   bool bDoLightingWarp, in sampler2D lightWarpSampler )
{
	float fResult;

	float NDotL = dot( worldNormal, lightDir );				// Unsaturated dot (-1 to 1 range)

	if ( bHalfLambert )
	{
		fResult = saturate(NDotL * 0.5 + 0.5);				// Scale and bias to 0 to 1 range

		if ( !bDoLightingWarp )
		{
			fResult *= fResult;								// Square
		}
	}
	else
	{
		fResult = saturate( NDotL );						// Saturate pure Lambertian term
		fResult = SoftenCosineTerm( fResult );				// For CS:GO
	}

	vec3 fOut = vec3( fResult);
	if ( bDoLightingWarp )
	{
		fOut = texture( lightWarpSampler, vec2(min(fResult,0.99f))).rgb;
	}

	#ifdef INV_GAMMA
		fOut= GammaToLinear(fOut);
	#endif

	return fOut;
}


vec3 PixelShaderGetLightVector( vec3 worldPos, PixelShaderLightInfo cLightInfo[3], int nLightIndex )
{
	if ( nLightIndex == 3 )
	{
		// Unpack light 3 from w components...
		vec3 vLight3Pos = vec3( cLightInfo[1].pos.w, cLightInfo[2].color.w, cLightInfo[2].pos.w );
		return normalize( vLight3Pos - worldPos );
	}
	else
	{
		return normalize( cLightInfo[nLightIndex].pos.xyz - worldPos );
	}
}


vec3 PixelShaderGetLightColor( PixelShaderLightInfo cLightInfo[3], int nLightIndex )
{
	if ( nLightIndex == 3 )
	{
		// Unpack light 3 from w components...
		return vec3( cLightInfo[0].color.w, cLightInfo[0].pos.w, cLightInfo[1].color.w );
	}
	else
	{
		return cLightInfo[nLightIndex].color.rgb;
	}
}

vec3 PixelShaderDoGeneralDiffuseLight( float fAtten, vec3 worldPos, vec3 worldNormal,
										 in vec3 NormalizeSampler,
										 vec3 vPosition, vec3 vColor, bool bHalfLambert,
										 bool bDoLightingWarp, in sampler2D lightWarpSampler )
{
	vec3 lightDir = normalize( vPosition - worldPos );
	return vColor * fAtten * DiffuseTerm( bHalfLambert, worldNormal, lightDir,
										  bDoLightingWarp, lightWarpSampler );
}


void CharacterSpecularAndRimTerms( vec3 vWorldNormal, vec3 vLightDir, float fSpecularExponent, vec3 vEyeDir,
						  bool bDoSpecularWarp, in sampler2D specularWarpSampler,
						  vec3 color, bool bDoRimLighting, float fRimExponent, float fWarpIndex,
						  bool bDoAnisotropy, float fAnisoAmount, float VdotT, float sVdotT, vec3 vTangent,
						  bool bDoRetroReflectivity, float fRetroReflectivityAmount, float fRetroReflectivityFresnel,

						  // Outputs
						  out vec3 specularLighting, out vec3 rimLighting, out float rimHalo )
{
	vec3 vHalfAngle = normalize( vEyeDir.xyz + vLightDir.xyz );
	float flNDotH = saturate( dot( vWorldNormal.xyz, vHalfAngle.xyz ) );
	float flNDotL = saturate( dot( vWorldNormal, vLightDir ) );
	specularLighting = vec3( 0.0f );

	if ( bDoAnisotropy )
	{
		float LdotT = dot( vLightDir, vTangent );
		float sLdotT = sqrt( 1 - LdotT * LdotT );

		float anisotropicSpecular = saturate( VdotT * LdotT + sVdotT * sLdotT );

		flNDotH = mix( flNDotH, anisotropicSpecular, fAnisoAmount );
	}

	// Optionally warp as function of scalar specular
	if ( bDoSpecularWarp )
	{
		#ifdef INV_U
			float specularU = clamp(1.0f - flNDotH,0.01f,0.99f);
		#else
			float specularU = clamp(flNDotH,0.01f,0.99f);
		#endif
		
		specularLighting = texture( specularWarpSampler, vec2( specularU , fWarpIndex ) ).rgb; //0.99f fix dark Dot
	// #ifdef INV_GAMMA
	// 	specularLighting = GammaToLinear(specularLighting);
	// #endif

	}
	else
	{
		specularLighting = vec3(pow( flNDotH, fSpecularExponent ));
	}
	// #ifdef INV_GAMMA
	// 	specularLighting = GammaToLinear(specularLighting);
	// #endif

	if ( bDoRetroReflectivity )
	{
		float flVDotL = saturate( dot( vEyeDir.xyz, vLightDir.xyz ) );
		specularLighting = mix( specularLighting, vec3(fRetroReflectivityFresnel * flVDotL * g_fRetroReflectivityBoost), vec3(fRetroReflectivityAmount) );
	}

	specularLighting *= pow( flNDotL, 0.5 );

	#ifdef INV_GAMMA
		specularLighting = GammaToLinear(specularLighting);
	#endif

	specularLighting *= color;													// Modulate with light color



	// Optionally do rim lighting
	rimLighting = vec3( 0.0, 0.0, 0.0 );
	rimHalo = 0;
	if ( bDoRimLighting )
	{
		float flNDotV = 1.0f - saturate( dot( vWorldNormal.xyz, vEyeDir.xyz ) );

		rimHalo  = flNDotH * flNDotL;
		rimHalo *= pow( flNDotV, fRimExponent );
		rimHalo *= pow( flNDotL, 0.5 );
		
		#ifdef INV_GAMMA
			rimHalo = GammaToLinear(rimHalo);
		#endif
		rimLighting = rimHalo * color;

	}
}

void CharacterDoSpecularLighting( vec3 worldPos, vec3 vWorldNormal, float fSpecularExponent, vec3 vEyeDir,
								  vec4 lightAtten, int nNumLights, PixelShaderLightInfo cLightInfo[3],
								  bool bDoSpecularWarp, in sampler2D specularWarpSampler,
								  bool bDoRimLighting, float fRimExponent, float flDirectShadow, float fWarpIndex,
								  bool bDoAnisotropy, float fAnisoAmount, float fAnisotropyAngle,
								  vec3 vTangent,
								  bool bDoRetroReflectivity, float fRetroReflectivityAmount, vec3 ambient,

								  // Outputs
								  out vec3 specularLighting, out vec3 rimLighting, out float rimHalo )
{
	specularLighting = rimLighting = vec3( 0.0f );
	rimHalo = 0.0f;
	vec3 localSpecularTerm, localRimTerm = vec3( 0.0f );
	float localRimHalo = 0.0f;
	float flVDotN = 0.0f;
	if ( bDoRetroReflectivity )
	{
		flVDotN = saturate( dot( vWorldNormal.xyz, vEyeDir.xyz ) );
		flVDotN = pow( flVDotN, g_fRetroReflectivityPower );
		float retroReflectivity = fRetroReflectivityAmount * flVDotN  * g_fRetroReflectivityBoost;
		#ifdef INV_GAMMA
			retroReflectivity = GammaToLinear(retroReflectivity);
		#endif
		specularLighting +=  retroReflectivity * ambient;

	}

	float VdotT = 1;
	float sVdotT = 1;
	if ( bDoAnisotropy )
	{

		VdotT = dot( vEyeDir, vTangent );
		sVdotT = sqrt( 1 - VdotT * VdotT );
	}

	if( nNumLights > 0 )
	{
		// First local light will always be forced to a directional light in CS:GO (see CanonicalizeMaterialLightingState() in shaderapidx8.cpp) - it may be completely black.
		CharacterSpecularAndRimTerms( vWorldNormal, PixelShaderGetLightVector( worldPos, cLightInfo, 0 ), fSpecularExponent, vEyeDir,
							 bDoSpecularWarp, specularWarpSampler, PixelShaderGetLightColor( cLightInfo, 0 ) * lightAtten[0],
							 bDoRimLighting, fRimExponent, fWarpIndex,
							 bDoAnisotropy, fAnisoAmount, VdotT, sVdotT, vTangent,
							 bDoRetroReflectivity, fRetroReflectivityAmount, flVDotN,
							 localSpecularTerm, localRimTerm, localRimHalo );


		specularLighting += localSpecularTerm * flDirectShadow;		// Accumulate specular and rim terms
		rimLighting += localRimTerm * flDirectShadow;
		rimHalo += localRimHalo;
	}

	if( nNumLights > 1 )
	{
		CharacterSpecularAndRimTerms( vWorldNormal, PixelShaderGetLightVector( worldPos, cLightInfo, 1 ), fSpecularExponent, vEyeDir,
							 bDoSpecularWarp, specularWarpSampler, PixelShaderGetLightColor( cLightInfo, 1 ) * lightAtten[1],
							 bDoRimLighting, fRimExponent, fWarpIndex,
							 bDoAnisotropy, fAnisoAmount, VdotT, sVdotT, vTangent,
							 bDoRetroReflectivity, fRetroReflectivityAmount, flVDotN,
							 localSpecularTerm, localRimTerm, localRimHalo );

		specularLighting += localSpecularTerm;		// Accumulate specular and rim terms
		rimLighting += localRimTerm;
		rimHalo += localRimHalo;
	}


	if( nNumLights > 2 )
	{
		CharacterSpecularAndRimTerms( vWorldNormal, PixelShaderGetLightVector( worldPos, cLightInfo, 2 ), fSpecularExponent, vEyeDir,
							 bDoSpecularWarp, specularWarpSampler, PixelShaderGetLightColor( cLightInfo, 2 ) * lightAtten[2],
							 bDoRimLighting, fRimExponent, fWarpIndex,
							 bDoAnisotropy, fAnisoAmount, VdotT, sVdotT, vTangent,
							 bDoRetroReflectivity, fRetroReflectivityAmount, flVDotN,
							 localSpecularTerm, localRimTerm, localRimHalo );

		specularLighting += localSpecularTerm;		// Accumulate specular and rim terms
		rimLighting += localRimTerm;
		rimHalo += localRimHalo;
	}

	if( nNumLights > 3 )
	{
		CharacterSpecularAndRimTerms( vWorldNormal, PixelShaderGetLightVector( worldPos, cLightInfo, 3 ), fSpecularExponent, vEyeDir,
							 bDoSpecularWarp, specularWarpSampler, PixelShaderGetLightColor( cLightInfo, 3 ) * lightAtten[3],
							 bDoRimLighting, fRimExponent, fWarpIndex,
							 bDoAnisotropy, fAnisoAmount, VdotT, sVdotT, vTangent,
							 bDoRetroReflectivity, fRetroReflectivityAmount, flVDotN,
							 localSpecularTerm, localRimTerm, localRimHalo );

		specularLighting += localSpecularTerm;		// Accumulate specular and rim terms
		rimLighting += localRimTerm;
		rimHalo += localRimHalo;
	}
}

vec3 PixelShaderDoLightingLinear( vec3 worldPos, vec3 worldNormal,
				   vec3 staticLightingColor, bool bStaticLight,
				   bool bAmbientLight, vec4 lightAtten, vec3 cAmbientCube,
				   in vec3 NormalizeSampler, int nNumLights, PixelShaderLightInfo cLightInfo[3], bool bHalfLambert,
				   bool bDoLightingWarp, in sampler2D lightWarpSampler, float flDirectShadow )
{
	vec3 linearColor = vec3(0.0f);

	if ( bStaticLight && !bAmbientLight )
	{
		linearColor += (staticLightingColor * cOverbright );
	}

	if ( bAmbientLight )
	{
		linearColor += (cAmbientCube);
	}


	if ( nNumLights > 0 )
	{
		// First local light will always be forced to a directional light in CS:GO (see CanonicalizeMaterialLightingState() in shaderapidx8.cpp) - it may be completely black.
		linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.x, worldPos, worldNormal, NormalizeSampler,
														 cLightInfo[0].pos.xyz, cLightInfo[0].color.rgb, bHalfLambert,
														 bDoLightingWarp, lightWarpSampler ) * flDirectShadow;
		if ( nNumLights > 1 )
		{
			linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.y, worldPos, worldNormal, NormalizeSampler,
															 cLightInfo[1].pos.xyz, cLightInfo[1].color.rgb, bHalfLambert,
															 bDoLightingWarp, lightWarpSampler );
			if ( nNumLights > 2 )
			{
				linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.z, worldPos, worldNormal, NormalizeSampler,
																 cLightInfo[2].pos.xyz, cLightInfo[2].color.rgb, bHalfLambert,
																 bDoLightingWarp, lightWarpSampler );
				if ( nNumLights > 3 )
				{
					// Unpack the 4th light's data from tight constant packing
					vec3 vLight3Color = vec3( cLightInfo[0].color.w, cLightInfo[0].pos.w, cLightInfo[1].color.w );
					vec3 vLight3Pos = vec3( cLightInfo[1].pos.w, cLightInfo[2].color.w, cLightInfo[2].pos.w );
					linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.w, worldPos, worldNormal, NormalizeSampler,
																	 vLight3Pos, vLight3Color, bHalfLambert,
																	 bDoLightingWarp, lightWarpSampler );
				}
			}
		}
	}

	return linearColor;
}
vec3 PixelShaderDoLighting( vec3 worldPos, vec3 worldNormal,
				   vec3 staticLightingColor, bool bStaticLight,
				   bool bAmbientLight, vec4 lightAtten, vec3 cAmbientCube,
				   in vec3 NormalizeSampler, int nNumLights, PixelShaderLightInfo cLightInfo[3],
				   bool bHalfLambert, bool bDoLightingWarp, in sampler2D lightWarpSampler, float flDirectShadow = 1.0f )
{
	vec3 linearColor = PixelShaderDoLightingLinear( worldPos, worldNormal, staticLightingColor,
													  bStaticLight, bAmbientLight, lightAtten,
													  cAmbientCube, NormalizeSampler, nNumLights, cLightInfo, bHalfLambert,
													  bDoLightingWarp, lightWarpSampler, flDirectShadow );

		// go ahead and clamp to the linear space equivalent of overbright 2 so that we match everything else.
//		linearColor = HuePreservingColorClamp( linearColor, pow( 2.0f, 2.2 ) );

	return linearColor;
}

vec4 computeLightAtten (vec4 lightAtten_input,int NUM_LIGHTS)
{
	vec4 lights = vec4(0.0);
	if (NUM_LIGHTS > 0)
		lights.x = lightAtten_input.x;
	if (NUM_LIGHTS > 1)
		lights.y = lightAtten_input.y;
	if (NUM_LIGHTS > 2)
		lights.z = lightAtten_input.z;
	if (NUM_LIGHTS > 3)
		lights.w = lightAtten_input.w;
	return lights;
}


void shade(V2F inputs)
{
	PropagateLights(inputs.position.xyz, r_lightBoost); // Bring in lights
	LocalVectors vectors 	= computeLocalFrame(inputs);
/*
//--------csgo Parameters ---------------------------------------------------//
*/
//combine shader parameters for csgo specific

    // vec4 g_vBounceTerms;
    // g_vBounceTerms.rgb                      = g_cBounce;
    // g_vBounceTerms.w                        = g_fAmbientBounceBoost;

    // vec4 g_vDiffuseModulation; //Tint diffuse texture by some inputs.

    // vec4 g_vDiffuseTerms;
    // g_vDiffuseTerms.x                       = g_fEnvmapLightScale;
    // g_vDiffuseTerms.y                       = g_fShadowSaturation;
    // g_vDiffuseTerms.z                       = g_fMetalness;
    // g_vDiffuseTerms.w                       = g_fRimLightAlbedo;

    // vec4 g_vPhongTerms;
    // g_vPhongTerms.x                         = g_fPhongBoost;
    // g_vPhongTerms.y                         = g_fPhongAlbedoBoost;
    // g_vPhongTerms.z                         = g_fPhongExponent;
    // g_vPhongTerms.w                         = g_fAnisotropyAmount;

    // vec4 g_vPhongTint_ShadowRimBoost;
    // g_vPhongTint_ShadowRimBoost.rgb         = g_cPhongTint;
    // g_vPhongTint_ShadowRimBoost.w           = g_fShadowRimBoost;

    // vec4 g_vEnvmapTerm;
    // g_vEnvmapTerm.x                         = g_vEnvmapLightScaleMin;
    // g_vEnvmapTerm.y                         = g_vEnvmapLightScaleMax;
    // g_vEnvmapTerm.z                         = g_fEnvmapContrast;
    // g_vEnvmapTerm.w                         = g_fEnvmapSaturation;

    // vec4 g_vRimTerms_SelfIllumTerms;
    // g_vRimTerms_SelfIllumTerms.x            = g_fRimLightExponent;
    // g_vRimTerms_SelfIllumTerms.y            = g_fRimLightBoost;
    // g_vRimTerms_SelfIllumTerms.z            = g_fSelfIllumBoost;
    // g_vRimTerms_SelfIllumTerms.w            = g_fWarpIndex;

    vec4 g_cRimLightTint_fRimHaloBoost;
    g_cRimLightTint_fRimHaloBoost.xyz       = g_cRimLightTint;
    g_cRimLightTint_fRimHaloBoost.w         = g_fRimHaloBoost * g_fRimHaloBoost ;

    vec4 g_vFakeRimTint_ShadowScale;
    g_vFakeRimTint_ShadowScale.rgb          = g_cFakeRimTint * g_fRimboost;
    g_vFakeRimTint_ShadowScale.w            = clamp( (1.0f - g_fShadowScale), 0.0f, 1.0f );

	vec4 cAmbient_fRimBoost; //Source Engine Input by TEXCOORD
    vec3 g_cAmbientCube 					= AmbientLight(vectors.normal); //cAmbientCube[6]
	cAmbient_fRimBoost.rgb 					= g_cAmbientCube;


	float fAmbientLightLum = dot( vec3( 0.299, 0.587, 0.114 ), g_cAmbientCube );
	cAmbient_fRimBoost.a = smoothstep( g_vShadowRimMax, g_vShadowRimMin, fAmbientLightLum ) * 10;//it is too small?mul 10

    vec3 g_vEyePos 							= camera_pos;
	bool bStaticLight 						= false;//there be no bStaticLight

	vec4 vWorldTangentS_vBounceCenterx; //Source Engine Input by TEXCOORD
	vec4 vWorldTangentT_vBounceCentery; //Source Engine Input by TEXCOORD
	vec4 vBounceCenterDir_vBounceCenterz; //Source Engine Input by TEXCOORD
	vWorldTangentS_vBounceCenterx.xyz 		= inputs.tangent.xyz ;	//vWorldTangentS
	vWorldTangentT_vBounceCentery.xyz		= inputs.bitangent.xyz ;//vWorldTangentT
	//vWorldTangentT_vBounceCentery.xyz 		= cross(inputs.normal.xyz,inputs.tangent.xyz);

	//vBounceCenterDir_vBounceCenterz.xyz 	= normalize( g_vBounceCenter - (uniform_camera_vp_matrix_inverse * vec4(inputs.position,1) ).xyz); //BounceCenter - app_position
	vBounceCenterDir_vBounceCenterz.xyz 	= normalize( g_vBounceCenter.xyz - inputs.position );
	//vBounceCenterDir_vBounceCenterz.xyz 	= normalize( ( vec4(inputs.position.xyz,1) * uniform_camera_view_matrix_it ).xyz - g_vBounceCenter.xyz );
	//vBounceCenterDir_vBounceCenterz.xyz 	= normalize( ( uniform_camera_view_matrix_it * vec4(g_vBounceCenter.zyx,1) ).xyz - inputs.position ); //substance bug,cant load uniform_camera_vp_matrix_inverse
	vBounceCenterDir_vBounceCenterz.xyz 	= is_perspective ? vBounceCenterDir_vBounceCenterz.xyz : vec3(0.0,0.0,1.0);

	vWorldTangentS_vBounceCenterx.w 		= g_vBounceCenter.x;
	vWorldTangentT_vBounceCentery.w 		= g_vBounceCenter.y;
	vBounceCenterDir_vBounceCenterz.w 		= g_vBounceCenter.z;


/*
initialize ps shader
*/
	vec3 vWorldPos 		= inputs.position.xyz;//source engine Using i.vWorldPos_projZ.xyz
    vec4 lightAtten;
	lightAtten.x 		= GetVertexAttenForLight( vWorldPos, 0 );
	lightAtten.y 		= GetVertexAttenForLight( vWorldPos, 1 );
	lightAtten.z 		= GetVertexAttenForLight( vWorldPos, 2 );
	lightAtten.w 		= GetVertexAttenForLight( vWorldPos, 3 );
	lightAtten 			= computeLightAtten(lightAtten,NUM_LIGHTS);
    //initialize SourceEngine shader common parameter
    //vec3 vWorldNormal 	= computeWSNormal(inputs.sparse_coord, inputs.tangent, inputs.bitangent, inputs.normal); //Use PBR computed WorldSpace Normal
	vec3 vWorldNormal 	= vectors.normal; //use engine computed WorldSpace Normal
	vec3 vEyeDir 		= is_perspective ? normalize(camera_pos - inputs.position) : -camera_dir; //2D View using face camera
	vec3 vReflection 	= 2 * vWorldNormal * dot( vWorldNormal, vEyeDir ) - vEyeDir;
	float fCSMShadow 	= getShadowFactor(); //use engine shadow
	vec3 NormalizeSampler 		= (light_main.xyz - inputs.position); //main_light : a vec4 indicating the position of the main light in the environment

/*
texture to value
*/

	float fSpecMask 			= sampleWithDefault(SpecularSampler       	, inputs.sparse_coord, 1.0f);
	float fEnvMask 				= sampleWithDefault(EnvMapSampler     		, inputs.sparse_coord, 1.0f);
	float fSelfIllumMask 		= sampleWithDefault(emissive_tex     		, inputs.sparse_coord, 0.0f);

    vec4 vBaseTextureSample;
    vBaseTextureSample.rgb      = getBaseColor(		BaseTextureSampler      , inputs.sparse_coord);
	vBaseTextureSample.a        = sampleWithDefault(OpacityAlphaSampler     , inputs.sparse_coord, 1.0);

    vec3 cBase 	= vBaseTextureSample.rgb ;
	float fAlpha = vBaseTextureSample.a ;

	if ( BASEALPHAPHONGMASK )
    {
		fSpecMask =  vBaseTextureSample.a;
		fAlpha = 1.0f;
    }

	if ( BASEALPHAENVMASK )
    {
		fEnvMask =  vBaseTextureSample.a;
		fAlpha = 1.0f;
    }

	if ( BASEALPHASELFILLUMMASK )
    {
		fSelfIllumMask =  vBaseTextureSample.a;
		fAlpha = 1.0f;
    }

    //mask1

	float fRimMask = 1.0f;
	float fMetalnessMask = 1.0f; //Dark Color mask
	float fPhongAlbedoMask = 0.0f; //Real Metalness Mask
	float fWarpIndex = 1.0f - g_fWarpIndex;

    vec4 vMasks1Params;
	vMasks1Params.r             = sampleWithDefault(RimlightSampler         , inputs.sparse_coord, 1.0);
	vMasks1Params.g             = sampleWithDefault(AlbedoTintSampler       , inputs.sparse_coord, 0.0);
	vMasks1Params.b             = sampleWithDefault(MetallicSampler         , inputs.sparse_coord, g_fMetalness);
	vMasks1Params.a             = sampleWithDefault(WarpIndexMaskSampler    , inputs.sparse_coord, g_fWarpIndex);
	vMasks1Params.a 			= clamp(1.0f - vMasks1Params.a,0.01f,0.99f); //Forced to Clampe Range(0-1)

	if ( MASKS1 )
	{
		fRimMask = vMasks1Params.r;
		fPhongAlbedoMask = vMasks1Params.g;
		fMetalnessMask = vMasks1Params.b;
		fWarpIndex = vMasks1Params.a;
	}
	else
        fMetalnessMask = g_fMetalness;
	float fFresnel,fAmbientReflectionMask;
    //take Fresnel Term with warp
	if ( FRESNELRANGESTEXTURE )
    {
		fFresnel = saturate( dot( vEyeDir, vWorldNormal ) );

		#ifdef INV_U
			float sampleFresnelU = clamp(1.0f - fFresnel,0.01f,0.99f);
		#endif

		#ifndef INV_U
			float sampleFresnelU = clamp(fFresnel,0.01f,0.99f);
		#endif

		vec3 vFresnelParams = texture ( FresnelRangesSampler, vec2( sampleFresnelU, fWarpIndex ) ).rgb;

		// #ifdef INV_GAMMA
		// vFresnelParams = GammaToLinear(vFresnelParams);
		// #endif

		fFresnel = vFresnelParams.y;
		fAmbientReflectionMask = vFresnelParams.z * fRimMask;
    }
	else
    {
		fFresnel = Fresnel( vWorldNormal, vEyeDir, g_vFresnelRanges );
		fAmbientReflectionMask = fFresnel * fRimMask;
    }

    //mask2

	float fShadowSaturationMask = 1.0f;
	float fAnisotropyAmount = g_fAnisotropyAmount;
	float fAnisotropyAngle = 1.57;

	float fRetroReflectivityMask = 0.0f;
	float fEnvmapLightScale = g_fEnvmapLightScale;


	vec4 vMasks2Params; //read Mask2 parameters in linear space
    vMasks2Params.r = sampleWithDefault(ShadowSaturationSampler         , inputs.sparse_coord, 1.0);
    vMasks2Params.g = sampleWithDefault(AnisotropyAngleSampler          , inputs.sparse_coord, g_fAnisotropyAmount);
    vMasks2Params.b = sampleWithDefault(LightScaleSampler               , inputs.sparse_coord, 0.0);
    vMasks2Params.a = sampleWithDefault(RetroreflectivitySampler        , inputs.sparse_coord, 1.0);

	float fAmbientReflectionBoost = sampleWithDefault(AmbientReflectionBoostSampler, inputs.sparse_coord, r_fAmbientBounceBoost < 0.99f ? r_fAmbientBounceBoost : 1.0f );
	if( r_fAmbientBounceBoost > 0.99f) fAmbientReflectionBoost *= r_fAmbientBounceBoost;
	// fAmbientReflectionBoost *= fAmbientReflectionBoost; //Is Squart Refleciton Power nessary ?

	if ( MASKS2 )
    {
		fShadowSaturationMask *= vMasks2Params.x;
		fAnisotropyAmount *= float( vMasks2Params.g > 0 );
		fAnisotropyAngle = vMasks2Params.g * M_PI; // M_PI come from shader API
		fEnvmapLightScale *= vMasks2Params.b;
		fRetroReflectivityMask = 1.0f - vMasks2Params.a; //Retro Reflection --> 0 .HdotN Reflection --> 1.
		// HdotN Reflection would not reflection anything on UnfacedLight Noraml Surface without light infomation.but Retro Reflection can!(VdotL)
    }
//input base lightColor
	vec3 linearLightColor = PixelShaderDoLighting( vWorldPos, vWorldNormal,
				ambient_color, bAmbient,
				bAmbientLight, lightAtten, g_cAmbientCube,
				NormalizeSampler, NUM_LIGHTS, g_cLightInfo, ( HALFLAMBERT ),
				bDoLightingWarp, LightingWarpSampler, fCSMShadow );//csgo chracter shader force disable lightWarp

/*
reflection & specular term
*/
	vec3 cSpecularLight = vec3( 0.0f );
	vec3 cRimLight = vec3( 0.0f );
	vec3 cAdditiveRimlight = vec3( 0.0f );
	vec3 vTangent = vec3( 0.0f );
	vec3 vReflectionEyeVec = vEyeDir; // may be use as : reflect(normal,eve);
	float fRimHalo = 0;

	if ( FAKERIM && is_perspective ) // Fake Rim Light is a Staic Top Front Light,but just getting the Rimlight term :D
	{
		vec3 localRimTerm, localSpecularTerm = vec3( 0.0f );
		float localRimHalo = 0;
		//vec3 vFakeRimDir = vWorldPos; //hold the projection position,instead of rim params.
		//vec3 vFakeRimDir =  normalize( (  vec4( g_vViewFakeRimDir, 1 ) * uniform_camera_view_matrix_it ).xyz );//Does it need mvp?
		vec3 vFakeRimDir =  normalize( g_vViewFakeRimDir );//it is a light dir
		vec3 cFakeRimColor = cAmbient_fRimBoost.rgb * cAmbient_fRimBoost.a * g_vFakeRimTint_ShadowScale.xyz;

		CharacterSpecularAndRimTerms( vWorldNormal, vFakeRimDir, 1.0f, vEyeDir,
								false, PhongWarpSampler, cFakeRimColor,
								true, g_fRimLightExponent, 0.0f,
								false, 0.0f, 0.0f, 0.0f, vec3( 0.0f ),
								false, 0.0f, 0.0f,
								localSpecularTerm, localRimTerm, localRimHalo );
		cAdditiveRimlight += localRimTerm * fRimMask;
		// TODO: add rim saturation here?
	}


	if ( ANISOTROPY )
	{
		vec3 vnWorldTangentS = normalize( cross( vWorldNormal, vWorldTangentT_vBounceCentery.xyz ) );
		vec3 vnWorldTangentT = normalize( cross( vWorldNormal, vnWorldTangentS ) );

		float cr, sr;
		sincos( fAnisotropyAngle, cr, sr );
		vTangent = normalize( cr * vnWorldTangentT + sr * vnWorldTangentS ); // blend TangentSpace Normal by AnisotropyAngle

		// re-normalize
		vec3 rvec = cross( vTangent, vWorldNormal.xyz );
		vec3 uvec = cross( vEyeDir, rvec );
		vec3 evec = normalize( cross( rvec, vTangent ) );

		vReflectionEyeVec = mix( vEyeDir, evec, fAnisotropyAmount );

	}


	if ( PHONG )
	{
		CharacterDoSpecularLighting( vWorldPos, vWorldNormal, g_fPhongExponent, vEyeDir,
										lightAtten, NUM_LIGHTS, g_cLightInfo,
										( PHONGWARPTEXTURE ), PhongWarpSampler,
										( RIMLIGHT ), g_fRimLightExponent, fCSMShadow, fWarpIndex,
										( ANISOTROPY ), fAnisotropyAmount, fAnisotropyAngle,
										vTangent,
										( ( MASKS2 ) && ( fRetroReflectivityMask > 0 ) ), fRetroReflectivityMask, cAmbient_fRimBoost.xyz,

										// Outputs
										cSpecularLight, cRimLight, fRimHalo );
		// #if ( FLASHLIGHT )
		// 	cSpecularLight += cSpecularFlashlight;
		// #endif

		// cSpecularLight = LinearToGamma(cSpecularLight);

		if ( RIMLIGHT && is_perspective) //Block 2DView Rimlight
		{
			float fRimModulation = g_fRimLightBoost * fRimMask;
			float fRimBoost = cAmbient_fRimBoost.w * g_fShadowRimBoost;
			fRimBoost += 1.0f;
			fRimModulation *= fRimBoost;
			cRimLight *=  fRimModulation;
		}

		float fPhongBoost = g_fPhongBoost;
		cSpecularLight *= fSpecMask;

		if ( MASKS1 )
		{
			fPhongBoost = mix( g_fPhongBoost, g_fPhongAlbedoBoost, fPhongAlbedoMask );
		}

		cSpecularLight *= fPhongBoost;
	}

	if ( ENVMAP )
	{
		vec3 cEnvmap = envSampleLOD (vReflection, 4);

		cEnvmap = saturateColor( cEnvmap, g_fEnvmapSaturation );

		// use fCSMShadow preview
		vec3 cEnvmapLight = fCSMShadow * saturate( ( ( linearLightColor + cAmbient_fRimBoost.xyz ) - g_vEnvmapLightScaleMin ) * g_vEnvmapLightScaleMax );
		cEnvmap = mix( cEnvmap, cEnvmap * cEnvmapLight , fEnvmapLightScale );

		cEnvmap = mix( cEnvmap, cEnvmap * cEnvmap, g_fEnvmapContrast );

		cEnvmap *= fEnvMask * g_cEnvmapTint;

		cSpecularLight += cEnvmap;
	}


	if ( PHONG || ENVMAP )
	{
		if ( MASKS2 )
		{
			fFresnel = mix( fFresnel, 1.0f, fRetroReflectivityMask );
		}
		cSpecularLight *= fFresnel;
	}

	if ( AMBIENTREFLECTION && is_perspective) //cancel AMBIENTREFLECTION in 2dView
	{
		
		vec3 cAmbientReflection = AmbientLight( vReflection );
		//vec3 cAmbientReflection = vec3(0,0,0);

		vec3 cAmbientLightColor = PixelShaderDoLighting( vWorldPos, vec3( 0.0f, 1.0f, 0.0f ),
				vec3( 0.0f, 0.0f, 0.0f), false,
				false, lightAtten, g_cAmbientCube,
				NormalizeSampler, min( NUM_LIGHTS, 1 ), g_cLightInfo, false,
				false, LightingWarpSampler, 1.0f );

		cAmbientReflection *= cAmbientLightColor;

		if ( USEBOUNCECOLOR )
		{
			vec3 vBounceCenter = vec3( vWorldTangentS_vBounceCenterx.w, vWorldTangentT_vBounceCentery.w, vBounceCenterDir_vBounceCenterz.w );

			vec3 linearLightBounceModulate = PixelShaderDoLighting( vBounceCenter, - vBounceCenterDir_vBounceCenterz.xyz,
				vec3( 0.0f, 0.0f, 0.0f), false,
				false, lightAtten, g_cAmbientCube,
				NormalizeSampler, min( NUM_LIGHTS, 1 ), g_cLightInfo, false,
				false, LightingWarpSampler, 1.0f );

			float fBounceTerm = saturate( dot( vWorldNormal, vBounceCenterDir_vBounceCenterz.xyz ) );
			// #ifdef INV_GAMMA
			// 	fBounceTerm = GammaToLinear(fBounceTerm);
			// #endif
			
			vec3 cBounce = g_cBounce * cAmbient_fRimBoost.xyz * linearLightBounceModulate;

			cAmbientReflection = mix( cAmbientReflection, cBounce, fBounceTerm );
		}

		cAmbientReflection *= (fAmbientReflectionBoost * fAmbientReflectionMask);

		#ifdef INV_GAMMA
			cAmbientReflection = GammaToLinear(cAmbientReflection);
		#endif

		cSpecularLight += cAmbientReflection;
	}


	float fRimLightAlbedo = g_fRimLightAlbedo;

	if ( MASKS1 )
	{
		cSpecularLight *= mix( vec3( 1.0f, 1.0f, 1.0f ), cBase, fPhongAlbedoMask );
		fRimLightAlbedo = g_fRimLightAlbedo * fPhongAlbedoMask;
	}

	cRimLight *= mix( g_cRimLightTint_fRimHaloBoost.xyz, cBase * fRimLightAlbedo, saturate( fRimLightAlbedo ) );

	float fShadowScale = saturate( g_vFakeRimTint_ShadowScale.w + cAmbient_fRimBoost.w ); // If we darken shadows to increase contrast, don't do it in very dark areas

	float lightIntensity = desaturateColor( linearLightColor + cRimLight ).r;
	float fShadeLevels = smoothstep( 0.3, 0.0, lightIntensity );

	if ( SHADOWSATURATION )
	{
		lightIntensity = desaturateColor( linearLightColor ).g;
		// dark-to-mid blend
		float fShadeLevelsDark = smoothstep( g_vShadowSaturationBounds.x, g_vShadowSaturationBounds.y, lightIntensity );
		// mid-to-light blend
		float fShadeLevelsLight = smoothstep( g_vShadowSaturationBounds.w, g_vShadowSaturationBounds.z, lightIntensity );
		if ( RIMLIGHT )
		{
			// don't just use linear lighting, make a nice saturated halo on the rimlight too
			float rimHalo = smoothstep( g_vRimHaloBounds.x, g_vRimHaloBounds.y, fRimHalo );
			rimHalo *= smoothstep( g_vRimHaloBounds.w, g_vRimHaloBounds.z, fRimHalo );
			rimHalo *= desaturateColor( cRimLight ).g;
			rimHalo *= g_cRimLightTint_fRimHaloBoost.w;
			lightIntensity += rimHalo;
			fShadeLevelsLight = fShadeLevelsLight +  rimHalo;
		}

		cBase = mix( cBase, saturateColor( cBase, g_fShadowSaturation ), saturate(fShadeLevelsDark * fShadeLevelsLight * fShadowSaturationMask) );
		cBase = mix( cBase, tintColor( cBase, g_cShadowTint.rgb, g_cShadowTint.a ) * fShadowScale, fShadeLevels );
	}
	else
		cBase = mix( cBase, tintColor( cBase, g_cShadowTint.rgb, g_cShadowTint.a ) * fShadowScale, fShadeLevels );

	// add ambient lighting to model
	if ( bAmbientLight )
		linearLightColor += (cAmbient_fRimBoost.rgb);
	if ( bAmbient && !bAmbientLight )
		linearLightColor += (ambient_color);

	cBase *= fMetalnessMask;

	vec3 finalColor = (( cBase * linearLightColor ) + cSpecularLight + cRimLight + cAdditiveRimlight);//shadow fix

	if ( SELFILLUM )
	{
		finalColor = mix( finalColor, vBaseTextureSample.rgb * ( 1.0f + g_fSelfIllumBoost ), fSelfIllumMask );//SelfIllumBoost can be 0,fix with ++1.0f
	}

	if ( bHasAlpha && ( fAlpha < 1.0f ) ) {
		alphaKill(inputs.sparse_coord);
	}

	finalColor *= g_DiffuseTint;

	diffuseShadingOutput(finalColor);

}

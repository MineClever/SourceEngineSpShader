//this shader is used for source engine games
//writed by MineClever aka 悍匪 Bandit
//if you find bugs , you can tell in Steam : https://steamcommunity.com/id/chinsesbandit/

/*
version:01
---------------------------
shader worked!

version:02
---------------------------
specular fix
rimlight mask fix

add aces hdr tone

version:02
---------------------------
fix for substance painter 2021

version:03
---------------------------
fix gamma view to match hlmv
*/

import lib-utils.glsl
import lib-env.glsl
import lib-normal.glsl
import lib-alpha.glsl
import lib-vectors.glsl
import lib-sparse.glsl
import lib-sampler.glsl


#define FIX_GAMMA_LOOK

#ifdef FIX_GAMMA_LOOK
	#define INV_GAMMA
	#define BOOST_BASE
#endif

#define cOverbright 2.0f
#ifdef BOOST_BASE
	#define BOOST_NUM 10
#else
	#define BOOST_NUM 1
#endif



/*
//--------Switch Parameters ---------------------------------------------------//
*/


//help obj to projection

//: param auto camera_view_matrix_it
uniform mat4 uniform_camera_view_matrix_it;

//: param auto camera_view_matrix
uniform mat4 uniform_camera_view_matrix;

//: state cull_face off

//: param custom { "default": false, "label": "HDR Lighting Clamp", "group": "Switch Parameters" }
uniform bool bHDR;

//: param custom { "default": false, "label": "Alpha Test", "group": "Switch Parameters" }
uniform bool bHasAlpha;

//: param custom { "default": true, "label": "CS:GO Mode", "group": "Switch Parameters" }
uniform bool CSGO;

//: param custom { "default": false, "label": "Use IBL Sun", "group": "Switch Parameters" }
uniform bool bIBL;

//: param custom { "default": false, "label": "Use IBL Sun Color", "group": "Switch Parameters" }
uniform bool bIBL_Color;

//: param custom { "default": false, "label": "Use IBL Ambient(Game)", "group": "Switch Parameters" }
uniform bool bAmbientLight;

//: param custom { "default": true, "label": "Use Ambient Color (hlmv)", "group": "Switch Parameters" }
uniform bool bAmbient;

//: param custom { "default": [0.2,0.2,0.2], "label": "Ambient Color", "widget": "color", "group": "Switch Parameters" }
uniform vec3 r_ambient_color;
#define ambient_color r_ambient_color


/*
//--------warptex Parameters ---------------------------------------------------//
*/

//: param custom { "default": [0.35, 0.75, 1.0], "label": "Fresnel Ranges", "group": "Warp Parameters" }
uniform vec3 g_FresnelRanges;

//: param custom { "default": false, "label": "Use Light Warp", "group": "Warp Parameters" }
uniform bool LIGHTWARPTEXTURE;

//: param custom { "default": false, "label": "Use Phong Warp", "group": "Warp Parameters" }
uniform bool PHONGWARPTEXTURE;


//-------- Diffuse Terms Parameters ---------------------------------------------------//


//: param custom { "default": false, "label": "Use HalfLambert", "group": "Diffuse Terms Parameters" }
uniform bool HALFLAMBERT;

//: param custom { "default": false, "label": "Use ColorTintMask", "group": "Diffuse Terms Parameters" }
uniform bool TINTMASKTEXTURE;

//: param custom { "default": 0.0, "label": "$InverseBlendTintByBaseAlpha", "min": 0.0, "max": 1.0,"step":1.0 , "group": "Diffuse Terms Parameters" }
uniform float g_fInverseBlendTintByBaseAlpha;

//: param custom { "default": [1.0,1.0,1.0], "label": "$color", "widget": "color", "group": "Diffuse Terms Parameters" }
uniform vec3 g_DiffuseTint;

//#define g_DiffuseTint LinearToGamma(r_DiffuseTint) //use gamma color to match game


//-------- Unique Phong Parameters ---------------------------------------------------//


//: param custom { "default": true, "label": "Use Phong Specular", "group": "Phong Parameters" }
uniform bool PHONG;

//: param custom { "default": true, "label": "Use Glossy as Expoent Map", "group": "Phong Parameters" }
uniform bool bPhongExpMap;

//: param custom { "default": true, "label": "Use Metal as AlbedoTint Map", "group": "Phong Parameters" }
uniform bool bSpecTintMap;

//: param custom { "default": 0.0, "label": "$BaseMapAlphaPhongMask", "min": 0.0, "max": 1.0,"step":1.0 , "group": "Phong Parameters" }
uniform float g_fBaseMapAlphaPhongMask;

//: param custom { "default": 0.0, "label": "$BaseMapLumPhongMask", "min": 0.0, "max": 1.0,"step":1.0 , "group": "Phong Parameters" }
uniform float g_fBaseLumPhongMask;

//: param custom { "default": 1.0, "label": "$PhongExponent", "min": 0.0, "max": 150.0, "group": "Phong Parameters" }
uniform float g_fSpecExp;

//: param custom { "default": 1.0, "label": "$PhongBoost", "min": 0.0, "max": 255.0, "group": "Phong Parameters" }
uniform float r_SpecularBoost;
#define g_SpecularBoost (r_SpecularBoost * BOOST_NUM)

//: param custom { "default": 1.0, "label": "$PhongAlbedoBoost", "min": 0.0, "max": 255.0, "group": "Phong Parameters" }
uniform float g_PhongAlbedoBoost;

//: param custom { "default": [1.0,1.0,1.0], "label": "$PhongTint(AlbedoTint)", "widget": "color", "group": "Phong Parameters" }
uniform vec3 g_SpecularTint;

//-------- Unique Envmap Parameters ---------------------------------------------------//

//: param custom { "default": false, "label": "Use Envmap(Opacity)", "group": "Envmap Parameters" }
uniform bool CUBEMAP;

//: param custom { "default": false, "label": "Use Envmap Mask", "group": "Envmap Parameters" }
uniform bool bUseEnvmask;

//: param custom { "default": false, "label": "$NormalMapAlphaEnvmapMask", "group": "Envmap Parameters" }
uniform bool g_bHasNormalMapAlphaEnvmapMask;

//: param custom { "default": false, "label": "$EnvmapmaskInTintmaskTexture", "group": "Envmap Parameters" }
uniform bool g_bEnvmapmaskInTintmaskTexture;

//: param custom { "default": 0.0, "label": "$EnvMapFresnel", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_fEnvMapFresnel;

//: param custom { "default": 0.0, "label": "$InvertPhongMask(Envmask)", "min": 0.0, "max": 1.0,"step":1.0 , "group": "Envmap Parameters" }
uniform float g_fInvertPhongMask;

//: param custom { "default": [1.0,1.0,1.0], "label": "$EnvmapTint", "widget": "color", "group": "Envmap Parameters" }
uniform vec3 g_vEnvmapTint;

//#define g_vEnvmapTint LinearToGamma(r_vEnvmapTint) //use gamma color to match game

//: param custom { "default": 1.0, "label": "$EnvmapLightScale", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_fEnvmapLightScale;

//: param custom { "default": 0.0, "label": "$EnvmapLightScaleMin", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_vEnvmapLightScaleMin;

//: param custom { "default": 1.0, "label": "$EnvmapLightScaleMax", "min": 0.0, "max": 1.0, "group": "Envmap Parameters" }
uniform float g_vEnvmapLightScaleMax;

#define ENV_MAP_SCALE 1.0f //set a offset factory


//-------- Rim Lighting Parameters ---------------------------------------------------//

//: param custom { "default": false, "label": "Use RimLight", "group": "RimLight Parameters" }
uniform bool RIMLIGHT;

//: param custom { "default": false, "label": "Use Rim Mask", "group": "RimLight Parameters" }
uniform bool RIMMASK;

//: param custom { "default": 5.0, "label": "$RimLightExponent", "min": 0.0, "max": 255.0, "group": "RimLight Parameters" }
uniform float g_RimExponent;

//: param custom { "default": 1.0, "label": "$RimLightBoost", "min": 0.0, "max": 255.0, "group": "RimLight Parameters" }
uniform float r_fRimBoost;
#define g_fRimBoost (r_fRimBoost * BOOST_NUM)


//-------- SelfIllum Parameters ---------------------------------------------------//



//: param custom { "default": false, "label": "Use SelfIllum", "group": "SelfIllum Parameters" }
uniform bool SELFILLUM;

//: param custom { "default": 0, "label": "$SelfIllumFresnel", "min": 0.0, "max": 1.0,"step":1, "group": "SelfIllum Parameters" }
uniform float SELFILLUMFRESNEL;

//: param custom { "default": [1.0,1.0,1.0], "label": "$SelfIllumTint", "widget": "color", "group": "SelfIllum Parameters" }
uniform vec3 g_SelfIllumTint;

//: param custom { "default": [0.0,1.0,1.0], "label": "$SelfIllumFresnelMinMaxExp", "min": 0.0, "max": 1.0, "group": "SelfIllum Parameters" }
uniform vec3 m_nSelfIllumFresnelMinMaxExp;


//-------- DetailTexture Parameters ---------------------------------------------------//

//: param custom { "default": false, "label": "Use DetailTexture", "group": "DetailTexture Parameters" }
uniform bool DETAILTEXTURE;

//: param custom { "default": 1.0, "label": "$DetailBlendFactor", "min": 0.0, "max":1.0, "group": "DetailTexture Parameters" }
uniform float g_DetailTextureBlendFactor;


//: param custom { "default": 0, "label": "$DetailBlendMode", "min": 0, "max": 7,"step":1 , "group": "DetailTexture Parameters" }
uniform int DETAIL_BLEND_MODE;


/*
//--------texure Sampler ---------------------------------------------------//
*/

//base texture

//: param auto channel_basecolor or channel_diffuse
uniform SamplerSparse BaseTextureSampler;

//: param auto channel_opacity
uniform SamplerSparse OpacityAlphaSampler;

//: param auto channel_specular or channel_specularlevel
uniform SamplerSparse SpecularSampler;

//: param auto channel_glossiness or channel_roughness
uniform SamplerSparse SpecExponentSampler;

//: param auto channel_emissive
uniform SamplerSparse SelfIllumMaskSampler;

//custom texture

//: param auto channel_metallic
uniform SamplerSparse TintMaskSampler;

//: param auto channel_user0
uniform SamplerSparse EnvMapSampler;

//: param auto channel_user1
uniform SamplerSparse ColorTintSampler;

//: param auto channel_user2
uniform SamplerSparse RimSampler;

//: param auto channel_user3
uniform SamplerSparse DetailSampler;

//: param auto channel_user4
uniform SamplerSparse DetailAlphaSampler;

//warpTexture

//: param custom { "default": "fresnelranges", "label": "LightWarp Tex", "usage": "texture", "group": "Warp Parameters" }
uniform sampler2D DiffuseWarpSampler;

//: param custom { "default": "fresnelranges", "label": "PhongWarp Tex", "usage": "texture", "group": "Warp Parameters" }
uniform sampler2D SpecularWarpSampler;



/*
//-------- Lights ----------------------------------------------------//
*/

//: param custom { "default": 1, "label": "Light Num", "min": 1, "max": 4, "group": "Lights Parameters" }
uniform int NUM_LIGHTS;

//: param custom { "default": 1.0, "label": "Light Lum Boost", "min": 1.0, "max": 1024.0, "group": "Lights Parameters" }
uniform float r_lightBoost;

//: param custom { "default": 2, "label": "Light Falloff Type.1=None,2=Distance,3=Expoent", "min": 1, "max": 3, "group": "Lights Parameters" }
uniform int r_DistanceType;

//: param custom { "default": 1.0, "label": "IBL Ambient Scale", "min": 0.0, "max": 5.0, "group": "Lights Parameters" }
uniform float r_AmbientScale;

//: param custom { "default": [0.0, 10.0, 10.0], "label": "Sun Position", "min": -10.0, "max": 10.0, "group": "Lights Parameters" }
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


//hlsl function

float saturate(float v)
{
  return clamp(v, 0.0, 1.0);
}

vec3 saturate(vec3 v)
{
  return clamp(v, 0.0, 1.0);
}

float rsqrt(float a)
{
  return pow(a, -0.5);
}

//custom Substance engine function

float sampleWithDefault(SamplerSparse sampler, SparseCoord coord, float defaultValue)
{
	vec2 value = textureSparse(sampler, coord).rg;
	return value.r + defaultValue * (1.0 - value.g);
}

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

//hdr_tone

// Base code sourced from Matt Pettineo's and Stephen Hill's work at https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

mat3 transpose(mat3 m) {
  return mat3(m[0][0], m[1][0], m[2][0],
              m[0][1], m[1][1], m[2][1],
              m[0][2], m[1][2], m[2][2]);
}
vec3 ACESFitted(vec3 color)
{
	mat3 ACESInputMat = mat3(
	    0.59719, 0.35458, 0.04823,
	    0.07600, 0.90834, 0.01566,
	    0.02840, 0.13383, 0.83777
	);
	 mat3 ACESOutputMat = mat3(
	     1.60475, -0.53108, -0.07367,
	    -0.10208,  1.10813, -0.00605,
	    -0.00327, -0.07276,  1.07602
	);
    color = transpose(ACESInputMat) * color;
    color = RRTAndODTFit(color);
    color = transpose(ACESOutputMat) * color;
    return clamp(color, 0.0, 1.0);
}


//Source Engine Function
vec4 GetFresnelSelfIllumParam (vec3 SelfIllumFresnelMinMaxExp)
{
	vec4 outputkey;
	float flMin = SelfIllumFresnelMinMaxExp.x;
	float flMax = SelfIllumFresnelMinMaxExp.y;
	float flExp = SelfIllumFresnelMinMaxExp.z;
	outputkey.y = ( flMax != 0.0f ) ? ( flMin / flMax ) : 0.0f; // Bias
	outputkey.x = 1.0f - outputkey.y; // Scale
	outputkey.z = flExp; //Exp
	outputkey.w = flMax; // Brightness
	return outputkey;
}

vec4 GetSelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost (vec3 g_SelfIllumTint,float g_DetailTextureBlendFactor,float g_PhongAlbedoBoost,bool DETAILTEXTURE)
{
	vec4 f;
	f.xyz = g_SelfIllumTint;

	if (DETAILTEXTURE)
	{
		f.w = g_DetailTextureBlendFactor;

	} else
		f.w = g_PhongAlbedoBoost;
	return f;
}

//source engine fresnel Ranges
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
	float fresnel = Fresnel2( vNormal, vEyeDir );						// Square
	return fresnel * fresnel;											// Square again for a more subtle look
}

vec3 CalcReflectionVectorUnnormalized( vec3 normal, vec3 eyeVector )
{
	// FIXME: might be better of normalizing with a normalizing cube map and
	// get rid of the dot( normal, normal )
	// compute reflection vector r = 2 * ((n dot v)/(n dot n)) n - v
	//  multiply all values through by N.N.  uniformly scaling reflection vector won't affect result
	//  since it is used in a cubemap lookup
	return (2.0*(dot( normal, eyeVector ))*normal) - (dot( normal, normal )*eyeVector);
}

//source engine function

// texture combining modes for combining base and detail/basetexture2
#define TCOMBINE_RGB_EQUALS_BASE_x_DETAILx2 0				// original mode
#define TCOMBINE_RGB_ADDITIVE 1								// base.rgb+detail.rgb*fblend
#define TCOMBINE_DETAIL_OVER_BASE 2
#define TCOMBINE_FADE 3										// straight fade between base and detail.
#define TCOMBINE_BASE_OVER_DETAIL 4                         // use base alpha for blend over detail
#define TCOMBINE_RGB_ADDITIVE_SELFILLUM 5                   // add detail color post lighting
#define TCOMBINE_RGB_ADDITIVE_SELFILLUM_THRESHOLD_FADE 6
#define TCOMBINE_MOD2X_SELECT_TWO_PATTERNS 7				// use alpha channel of base to select between mod2x channels in r+a of detail
#define TCOMBINE_MULTIPLY 8
#define TCOMBINE_MASK_BASE_BY_DETAIL_ALPHA 9                // use alpha channel of detail to mask base
#define TCOMBINE_SSBUMP_BUMP 10								// use detail to modulate lighting as an ssbump
#define TCOMBINE_SSBUMP_NOBUMP 11					// detail is an ssbump but use it as an albedo. shader does the magic here - no user needs to specify mode 11
#define TCOMBINE_NONE 12

vec4 TextureCombine( vec4 baseColor, vec4 detailColor, int combine_mode,
					   float fBlendFactor )
{
	if ( combine_mode == TCOMBINE_MOD2X_SELECT_TWO_PATTERNS)
	{
		vec3 dc=vec3(mix(detailColor.r,detailColor.a, baseColor.a));
		baseColor.rgb*=mix(vec3(1,1,1),2.0f*dc,vec3(fBlendFactor));
	}
	if ( combine_mode == TCOMBINE_RGB_EQUALS_BASE_x_DETAILx2)
		baseColor.rgb*=mix(vec3(1,1,1),2.0f*detailColor.rgb,vec3(fBlendFactor));
	if ( combine_mode == TCOMBINE_RGB_ADDITIVE )
 		baseColor.rgb += fBlendFactor * detailColor.rgb;
	if ( combine_mode == TCOMBINE_DETAIL_OVER_BASE )
	{
		float fblend=fBlendFactor * detailColor.a;
		baseColor.rgb = mix( baseColor.rgb, detailColor.rgb, vec3(fblend) );
	}
	if ( combine_mode == TCOMBINE_FADE )
	{
		baseColor = mix( baseColor, detailColor, vec4(fBlendFactor));
	}
	if ( combine_mode == TCOMBINE_BASE_OVER_DETAIL )
	{
		float fblend=fBlendFactor * (1-baseColor.a);
		baseColor.rgb = mix( baseColor.rgb, detailColor.rgb, vec3(fblend) );
		baseColor.a = detailColor.a;
	}
	if ( combine_mode == TCOMBINE_MULTIPLY )
	{
		baseColor = mix( baseColor, baseColor*detailColor, vec4(fBlendFactor));
	}

	if (combine_mode == TCOMBINE_MASK_BASE_BY_DETAIL_ALPHA )
	{
		baseColor.a = mix( baseColor.a, baseColor.a*detailColor.a, fBlendFactor );
	}
	if ( combine_mode == TCOMBINE_SSBUMP_NOBUMP )
	{
		baseColor.rgb = baseColor.rgb * dot( detailColor.rgb, vec3(2.0f/3.0f) );
	}
	return baseColor;
}

//lighting function

struct PixelShaderlightinfo
{
	vec4 color;
	vec4 pos;
    vec4 atten;//x=1,y=d,z=d*d
};

PixelShaderlightinfo lightinfo[3];

void PropagateLights(vec3 worldPos) //there are all light uniform parameter ,it is easy to load all set
{
    // First local light will always be forced to a directional light in CS:GO

    //encode lights with gamma
	vec3 temp_color,temp_pos;
	if (bIBL)
	{
		lightinfo[0].pos.xyz 	= (light_main.xyz + worldPos);
		if (!bIBL_Color)
			lightinfo[0].color.rgb   = (SunlightColor_0 * r_lightBoost);
		else
			temp_color 				= envIrradiance(light_main.xyz);
			lightinfo[0].color.rgb	= ( temp_color * r_lightBoost );

	} else
	{
		lightinfo[0].pos.xyz     = (Sunlight_0 + worldPos);
		lightinfo[0].color.rgb   = (SunlightColor_0 * r_lightBoost);
	}

	lightinfo[1].color.rgb   = (lightColor_1 * r_lightBoost);
	lightinfo[1].pos.xyz     = (lightPosition_1);
	lightinfo[2].color.rgb   = (lightColor_2 * r_lightBoost);
	lightinfo[2].pos.xyz     = (lightPosition_2);

	temp_color = (lightColor_3 * r_lightBoost);
	temp_pos = (lightPosition_3);
    //pos_3
	lightinfo[1].pos.w       = temp_pos.x;
	lightinfo[2].color.w     = temp_pos.y;
    lightinfo[2].pos.w       = temp_pos.z;
    //color_3
	lightinfo[0].color.w     = temp_color.r;
    lightinfo[0].pos.w       = temp_color.g;
    lightinfo[1].color.w     = temp_color.b;

	lightinfo[0].atten = vec4(1.0,0.0,0.0,1.0);
	lightinfo[1].atten = vec4(1.0,0.0,0.0,0.0);
	lightinfo[2].atten = vec4(1.0,0.0,0.0,0.0);

	if (r_DistanceType == 1)
	{
		lightinfo[0].atten = vec4(1.0,0.0,0.0,1.0);
		lightinfo[1].atten = vec4(1.0,0.0,0.0,0.0);
		lightinfo[2].atten = vec4(1.0,0.0,0.0,0.0);
	}
	if (r_DistanceType == 2)
	{
		lightinfo[0].atten = vec4(0.0,1.0,0.0,0.0);
		lightinfo[1].atten = vec4(0.0,1.0,0.0,1.0);
		lightinfo[2].atten = vec4(0.0,1.0,0.0,0.0);
	}
	if (r_DistanceType == 3)
	{
		lightinfo[0].atten = vec4(0.0,0.0,1.0,0.0);
		lightinfo[1].atten = vec4(0.0,0.0,1.0,0.0);
		lightinfo[2].atten = vec4(0.0,0.0,1.0,1.0);
	}
}


float VertexAttenInternal( vec3 worldPos, int lightNum ) //force set light as a point light for substance
{
	float result = 0.0f;
	vec3 lightDir;
	// Get light direction
	if ( lightNum == 3 )
	{
		// Unpack light 3 from w components...
		vec3 vLight3Pos = vec3( lightinfo[1].pos.w, lightinfo[2].color.w, lightinfo[2].pos.w );
		lightDir = ( vLight3Pos - worldPos );
	} else
		lightDir = lightinfo[lightNum].pos.xyz - worldPos;

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
        flDistanceAtten = 1.0f / dot( vec3(lightinfo[0].atten.w,lightinfo[1].atten.w,lightinfo[2].atten.w), vDist );
	else flDistanceAtten = 1.0f / dot( lightinfo[lightNum].atten.xyz, vDist );

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
	//return pow( flDot, SOFTEN_COSINE_EXP );
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
        if (CSGO) //global Bool,easy use
        {
		fResult = SoftenCosineTerm( fResult );
        }			// For CS:GO
	}

	vec3 fOut = vec3( fResult);
	if ( bDoLightingWarp )
	{
		float U = clamp(fResult,0.01,0.99);
		fOut = (texture( lightWarpSampler, vec2( U, 1 - U )).rgb);
	}

	#ifdef INV_GAMMA
		fOut = GammaToLinear(fOut);
	#endif

	return fOut;
}


vec3 PixelShaderGetLightVector( vec3 worldPos, PixelShaderlightinfo clightinfo[3], int nLightIndex )
{
	if ( nLightIndex == 3 )
	{
		// Unpack light 3 from w components...
		vec3 vLight3Pos = vec3( clightinfo[1].pos.w, clightinfo[2].color.w, clightinfo[2].pos.w );
		return normalize( vLight3Pos - worldPos );
	}
	else
	{
		return normalize( clightinfo[nLightIndex].pos.xyz - worldPos );
	}
}

vec3 PixelShaderGetLightColor( PixelShaderlightinfo clightinfo[3], int nLightIndex )
{
	if ( nLightIndex == 3 )
	{
		// Unpack light 3 from w components...
		return vec3( clightinfo[0].color.w, clightinfo[0].pos.w, clightinfo[1].color.w );
	}
	else
	{
		return clightinfo[nLightIndex].color.rgb;
	}
}

vec3 PixelShaderDoGeneralDiffuseLight( float fAtten, vec3 worldPos, vec3 worldNormal,
										 in vec3 NormalizeSampler,
										 vec3 vPosition, vec3 vColor, bool bHalfLambert,
										 bool bDoLightingWarp, in sampler2D lightWarpSampler )
{
	vec3 lightDir = vec3(0.0);

		lightDir = normalize( vPosition - worldPos );

	return vColor * fAtten * DiffuseTerm( bHalfLambert, worldNormal, lightDir,
										  bDoLightingWarp, lightWarpSampler );
}//fixed lighting function

void SpecularAndRimTerms( vec3 vWorldNormal, vec3 vLightDir, float fSpecularExponent, vec3 vEyeDir,
						  bool bDoSpecularWarp, in sampler2D specularWarpSampler, float fFresnel,
						  vec3 color, bool bDoRimLighting, float fRimExponent,

						  // Outputs
						  out vec3 specularLighting, out vec3 rimLighting )
{
	vec3 vHalfAngle = normalize( vEyeDir.xyz + vLightDir.xyz );
	float flNDotH = saturate( dot( vWorldNormal.xyz, vHalfAngle.xyz ) );
	specularLighting = vec3(pow( flNDotH, fSpecularExponent )); // Raise to specular exponent

	// Optionally warp as function of scalar specular and fresnel
	if ( bDoSpecularWarp )
	{
		vec2 UV = vec2(clamp(specularLighting.x,0.01,0.99) , 1 - fFresnel);
		specularLighting *= texture( specularWarpSampler, UV ).rgb; // Sample at { (N.H)^k, fresnel }
	}

	if (CSGO) // global , easy use
	{
		specularLighting *= pow( saturate( dot( vWorldNormal, vLightDir ) ), 0.5 ); // Mask with N.L raised to a power
	}
	else
		specularLighting *= saturate( dot( vWorldNormal, vLightDir )); // Mask with N.L

	#ifdef INV_GAMMA
		specularLighting = GammaToLinear(specularLighting);
	#endif
	specularLighting *= color;													// Modulate with light color

	// Optionally do rim lighting
	rimLighting = vec3( 0.0, 0.0, 0.0 );
	if ( bDoRimLighting )
	{
		rimLighting  = vec3(pow( flNDotH, fRimExponent ));					// Raise to rim exponent
		rimLighting *= saturate(dot( vWorldNormal, vLightDir ));		// Mask with N.L
	#ifdef INV_GAMMA
		rimLighting = GammaToLinear(rimLighting);
	#endif
		rimLighting *= color;											// Modulate with light color
	}
}


void PixelShaderDoSpecularLight( vec3 vWorldPos, vec3 vWorldNormal, float fSpecularExponent, vec3 vEyeDir,
								 float fAtten, vec3 vLightColor, vec3 vLightDir,
								 bool bDoSpecularWarp, in sampler2D specularWarpSampler, float fFresnel,
								 bool bDoRimLighting, float fRimExponent,

								 // Outputs
								 out vec3 specularLighting, out vec3 rimLighting )
{
	// Compute Specular and rim terms
	SpecularAndRimTerms( vWorldNormal, vLightDir, fSpecularExponent, vEyeDir,
						 bDoSpecularWarp, specularWarpSampler, fFresnel, vLightColor * fAtten,
						 bDoRimLighting, fRimExponent, specularLighting, rimLighting );
}

vec3 PixelShaderDoLightingLinear( vec3 worldPos, vec3 worldNormal,
				   vec3 staticLightingColor, bool bStaticLight,
				   bool bAmbientLight, vec4 lightAtten, vec3 cAmbientCube,
				   in vec3 NormalizeSampler, int nNumLights, PixelShaderlightinfo clightinfo[3], bool bHalfLambert,
				   bool bDoLightingWarp, in sampler2D lightWarpSampler, float flDirectShadow )
{
	vec3 linearColor = vec3(0.0f);

	if ( bStaticLight && !bAmbientLight)
	{
		linearColor += GammaToLinear(staticLightingColor * cOverbright );
	}

	if ( bAmbientLight )
	{
		linearColor += (cAmbientCube)*r_AmbientScale;
	}


	if ( nNumLights > 0 )
	{
		// First local light will always be forced to a directional light in CS:GO (see CanonicalizeMaterialLightingState() in shaderapidx8.cpp) - it may be completely black.
		linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.x, worldPos, worldNormal, NormalizeSampler,
														 clightinfo[0].pos.xyz, clightinfo[0].color.rgb, bHalfLambert,
														 bDoLightingWarp, lightWarpSampler ) * flDirectShadow;
		if ( nNumLights > 1 )
		{
			linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.y, worldPos, worldNormal, NormalizeSampler,
															 clightinfo[1].pos.xyz, clightinfo[1].color.rgb, bHalfLambert,
															 bDoLightingWarp, lightWarpSampler );
			if ( nNumLights > 2 )
			{
				linearColor += PixelShaderDoGeneralDiffuseLight( lightAtten.z, worldPos, worldNormal, NormalizeSampler,
																 clightinfo[2].pos.xyz, clightinfo[2].color.rgb, bHalfLambert,
																 bDoLightingWarp, lightWarpSampler );
				if ( nNumLights > 3 )
				{
					// Unpack the 4th light's data from tight ant packing
					vec3 vLight3Color = vec3( clightinfo[0].color.w, clightinfo[0].pos.w, clightinfo[1].color.w );
					vec3 vLight3Pos = vec3( clightinfo[1].pos.w, clightinfo[2].color.w, clightinfo[2].pos.w );
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
				   in vec3 NormalizeSampler, int nNumLights, PixelShaderlightinfo clightinfo[3],
				   bool bHalfLambert, bool bDoLightingWarp, in sampler2D lightWarpSampler, float flDirectShadow)
{
	flDirectShadow = 1.0f;
	vec3 linearColor = PixelShaderDoLightingLinear( worldPos, worldNormal, staticLightingColor,
													  bStaticLight, bAmbientLight, lightAtten,
													  cAmbientCube, NormalizeSampler, nNumLights, clightinfo, bHalfLambert,
													  bDoLightingWarp, lightWarpSampler, flDirectShadow );

		// go ahead and clamp to the linear space equivalent of overbright 2 so that we match everything else.
//		linearColor = HuePreservingColorClamp( linearColor, pow( 2.0f, 2.2 ) );

	return linearColor;
}


void PixelShaderDoSpecularLighting(  vec3 worldPos,  vec3 worldNormal,  float fSpecularExponent,  vec3 vEyeDir,
									 vec4 lightAtten,  int nNumLights, PixelShaderlightinfo clightinfo[3],
									 bool bDoSpecularWarp, in sampler specularWarpSampler, float fFresnel,
									 bool bDoRimLighting,  float fRimExponent,  float flDirectShadow,

									// Outputs
									out vec3 specularLighting, out vec3 rimLighting )
{
	specularLighting = rimLighting = vec3( 0.0f, 0.0f, 0.0f );
	vec3 localSpecularTerm, localRimTerm;

	if( nNumLights > 0 )
	{
		// First local light will always be forced to a directional light in CS:GO (see CanonicalizeMaterialLightingState() in shaderapidx8.cpp) - it may be completely black.
		PixelShaderDoSpecularLight( worldPos, worldNormal, fSpecularExponent, vEyeDir,
									lightAtten.x, PixelShaderGetLightColor( clightinfo, 0 ),
									PixelShaderGetLightVector( worldPos, clightinfo, 0 ),
									bDoSpecularWarp, specularWarpSampler, fFresnel,
									bDoRimLighting, fRimExponent,
									localSpecularTerm, localRimTerm );

		specularLighting += localSpecularTerm * flDirectShadow;		// Accumulate specular and rim terms
		rimLighting += localRimTerm * flDirectShadow;
	}

	if( nNumLights > 1 )
	{
		PixelShaderDoSpecularLight( worldPos, worldNormal, fSpecularExponent, vEyeDir,
									lightAtten.y, PixelShaderGetLightColor( clightinfo, 1 ),
									PixelShaderGetLightVector( worldPos, clightinfo, 1 ),
									bDoSpecularWarp, specularWarpSampler, fFresnel,
									bDoRimLighting, fRimExponent,
									localSpecularTerm, localRimTerm );

		specularLighting += localSpecularTerm;		// Accumulate specular and rim terms
		rimLighting += localRimTerm;
	}


	if( nNumLights > 2 )
	{
		PixelShaderDoSpecularLight( worldPos, worldNormal, fSpecularExponent, vEyeDir,
									lightAtten.z, PixelShaderGetLightColor( clightinfo, 2 ),
									PixelShaderGetLightVector( worldPos, clightinfo, 2 ),
									bDoSpecularWarp, specularWarpSampler, fFresnel,
									bDoRimLighting, fRimExponent,
									localSpecularTerm, localRimTerm );

		specularLighting += localSpecularTerm;		// Accumulate specular and rim terms
		rimLighting += localRimTerm;
	}

	if( nNumLights > 3 )
	{
		PixelShaderDoSpecularLight( worldPos, worldNormal, fSpecularExponent, vEyeDir,
									lightAtten.w, PixelShaderGetLightColor( clightinfo, 3 ),
									PixelShaderGetLightVector( worldPos, clightinfo, 3 ),
									bDoSpecularWarp, specularWarpSampler, fFresnel,
									bDoRimLighting, fRimExponent,
									localSpecularTerm, localRimTerm );

		specularLighting += localSpecularTerm;		// Accumulate specular and rim terms
		rimLighting += localRimTerm;
	}

}

vec3 TextureCombinePostLighting( vec3 lit_baseColor, vec4 detailColor, int combine_mode,
								   float fBlendFactor )
{
	if ( combine_mode == TCOMBINE_RGB_ADDITIVE_SELFILLUM )
 		lit_baseColor += fBlendFactor * detailColor.rgb;
	if ( combine_mode == TCOMBINE_RGB_ADDITIVE_SELFILLUM_THRESHOLD_FADE )
	{
 		// fade in an unusual way - instead of fading out color, remap an increasing band of it from
 		// 0..1
		if ( fBlendFactor > 0.5f)
			lit_baseColor += min(vec3(1), (1.0f/fBlendFactor)*max(vec3(0.0), detailColor.rgb-vec3(1-fBlendFactor) ) );
		else
			lit_baseColor += 2*fBlendFactor*2*max(vec3(0), detailColor.rgb-vec3(0.5f));
	}
	return lit_baseColor;
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

	//all in Camera Space
	PropagateLights(inputs.position.xyz); // Bring in lights
	LocalVectors vectors 	= computeLocalFrame(inputs);
	vec3 vWorldPos 			= inputs.position.xyz;
	//vec3 vWorldNormal   	= inputs.normal.xyz;
	//vec3 vWorldBinormal 	= inputs.bitangent.xyz;
    vec4 lightAtten;
	lightAtten.x 		= GetVertexAttenForLight( vWorldPos, 0 );
	lightAtten.y 		= GetVertexAttenForLight( vWorldPos, 1 );
	lightAtten.z 		= GetVertexAttenForLight( vWorldPos, 2 );
	lightAtten.w 		= GetVertexAttenForLight( vWorldPos, 3 );
	lightAtten 			= computeLightAtten(lightAtten,NUM_LIGHTS);
    vec4 vLightAtten        = lightAtten;

/*
//--------csgo Parameters ---------------------------------------------------//
*/
//combine shader parameters for csgo specific

	vec4 g_SelfIllumScaleBiasExpBrightness = GetFresnelSelfIllumParam(m_nSelfIllumFresnelMinMaxExp);

    vec4 g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost = GetSelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost ( g_SelfIllumTint, g_DetailTextureBlendFactor, g_PhongAlbedoBoost, DETAILTEXTURE);

    vec4 g_DiffuseModulation;
    g_DiffuseModulation.rgb = g_DiffuseTint;
    g_DiffuseModulation.a = 1.0;

    vec3 NormalizeRandRotSampler = light_main.xyz; //main_light : a vec4 indicating the position of the main light in the environment

    bool  PHONG_HALFLAMBERT     = HALFLAMBERT;

    if (PHONG && CSGO) // for csgo,Shader will force disable halflambert when Phong On
    {
       PHONG_HALFLAMBERT = false;//$phongforcedisabehalflambert
    }


    vec4 baseColor;
    baseColor.rgb   = getBaseColor(		BaseTextureSampler      , inputs.sparse_coord); //SPainter color in Engine Gamma space
    baseColor.a     = sampleWithDefault(OpacityAlphaSampler     , inputs.sparse_coord, 1.0);

	float fAlpha = baseColor.a;


	vec4 detailColor;
	if ( DETAILTEXTURE )
	{
		detailColor.rgb     = getBaseColor( DetailSampler, inputs.sparse_coord );
        detailColor.a       = sampleWithDefault(DetailAlphaSampler     , inputs.sparse_coord, 1.0);
		baseColor           = TextureCombine( baseColor, detailColor, DETAIL_BLEND_MODE, g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost.w );
	}

	vec3 lumCoefficients = { 0.3, 0.59, 0.11 };
	float baseLum = dot( baseColor.rgb, lumCoefficients );

	float fSpecMask = sampleWithDefault(SpecularSampler     , inputs.sparse_coord, 1.0);
	fSpecMask = mix( fSpecMask, baseColor.a, g_fBaseMapAlphaPhongMask );
	fSpecMask = mix( fSpecMask, baseLum, g_fBaseLumPhongMask );

	// We need a normal if we're doing any lighting
	vec3 worldSpaceNormal               = computeWSNormal(inputs.sparse_coord, inputs.tangent, inputs.bitangent, inputs.normal);
    vec3 vEyeDir                        = is_perspective ? normalize(camera_pos - inputs.position) :-camera_dir;
    vec3 cAmbientCube 					= envIrradiance(worldSpaceNormal); //cAmbientCube[6] //hdr color in linear
	cAmbientCube 						= bHDR?ACESFitted(cAmbientCube):cAmbientCube;

	float fFresnelRanges               	= Fresnel( worldSpaceNormal, vEyeDir, g_FresnelRanges );

	vec3 diffuseLighting                = vec3( 0.0f, 0.0f, 0.0f );
	vec3 envMapColor                    = vec3( 0.0f, 0.0f, 0.0f );

    float flDirectionalShadow           = getShadowFactor(); //get shadow from Substance Engine


    //start lighting !
    diffuseLighting = PixelShaderDoLighting( vWorldPos, worldSpaceNormal,
        ambient_color, bAmbient, bAmbientLight, vLightAtten,
        cAmbientCube, NormalizeRandRotSampler, NUM_LIGHTS, lightinfo, PHONG_HALFLAMBERT,
        LIGHTWARPTEXTURE, DiffuseWarpSampler, flDirectionalShadow );

    //float g_fInvertPhongMask = saturate(1 - fSpecMask);
	//diffuseLighting = bHDR?ACESFitted(diffuseLighting):diffuseLighting;


	float fSpecExp = g_fSpecExp;
    vec4 vSpecExpMap;
    vSpecExpMap.r = sampleWithDefault(SpecExponentSampler	, inputs.sparse_coord, 0.0);
    vSpecExpMap.g = sampleWithDefault(TintMaskSampler		, inputs.sparse_coord, 0.0);
    vSpecExpMap.b = 0.0;
    vSpecExpMap.a = sampleWithDefault(RimSampler     		, inputs.sparse_coord, 1.0);

    if( CUBEMAP )
    {
        vec3 vReflect = CalcReflectionVectorUnnormalized( worldSpaceNormal, vEyeDir );
        envMapColor = ENV_MAP_SCALE * (envSampleLOD (vReflect, 3).rgb) * g_vEnvmapTint.rgb;//ENV_MAP_SCALE to do? //hdr color in linear space

		// Optionally apply lightScale to envmap
		vec3 cEnvmapLight = saturate( ( ( diffuseLighting ) - g_vEnvmapLightScaleMin ) * g_vEnvmapLightScaleMax ) * flDirectionalShadow;//shadow fix
		envMapColor = mix( envMapColor, envMapColor * cEnvmapLight, g_fEnvmapLightScale );

        // Optionally apply Fresnel to envmap
        envMapColor = mix( envMapColor, fFresnelRanges * envMapColor, g_fEnvMapFresnel );

		//Tell engine which mask used as envmask,had it change render behave?
        float fEnvMapMask = baseColor.a;

		if (CSGO)
			fAlpha = 1.0;

		if ( bUseEnvmask )
		{
			fEnvMapMask = sampleWithDefault(EnvMapSampler, inputs.sparse_coord, 1.0);
			fAlpha = baseColor.a;
		}

		if ( g_bEnvmapmaskInTintmaskTexture )
		{
			fEnvMapMask = vSpecExpMap.r;
			fAlpha = baseColor.a;
		}

		if (g_bHasNormalMapAlphaEnvmapMask)
			fAlpha = baseColor.a;


        // Mask is either base map alpha or the same as the spec mask which can come from base map, normal map, or spec exponet map
        if ( SELFILLUMFRESNEL == 1.0 ) // This is to match the 2.0 version of vertexlitgeneric
        {
            fEnvMapMask = mix( fEnvMapMask, g_fInvertPhongMask, float( g_bHasNormalMapAlphaEnvmapMask ) );
        }
        else
        {
            fEnvMapMask = mix( fEnvMapMask, fSpecMask, float( g_bHasNormalMapAlphaEnvmapMask ) );
        }
        envMapColor *= mix( fEnvMapMask, 1-fEnvMapMask, g_fInvertPhongMask );
    }


	float fSpecExpMap = vSpecExpMap.r;
	float fRimMask = 0.0;
	float g_RimMaskControl = 0.0;
	if ( RIMMASK && is_perspective )
	{
		g_RimMaskControl = 1.0;
		//g_RimMaskControl = ( fAlpha < 1f ) ? 0.0:1.0;
	}

    fRimMask = mix( 1.0f, vSpecExpMap.a, g_RimMaskControl );				// Select rim mask

	if ( bPhongExpMap )
	{
		if (CSGO)
			fSpecExp = 1.0f - fSpecExpMap + 150.0f * fSpecExpMap;
		else
			fSpecExp = 1.0f + 149.0f * fSpecExpMap;//source-skd-2013
	}

	vec3 vSpecularTint = vec3(1.0f);

	if ( bSpecTintMap )
	{
		if ( DETAILTEXTURE && CSGO )
			vSpecularTint = g_SpecularBoost * mix( vec3( 1.0f ), baseColor.rgb, vec3(vSpecExpMap.g) );
		else
		{
			if ( CSGO )
				vSpecularTint = mix( vec3(g_SpecularBoost), g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost.w * baseColor.rgb, vec3(vSpecExpMap.g) );
			else
				vSpecularTint = g_SpecularBoost * mix( vec3(1.0f), baseColor.rgb, vec3(vSpecExpMap.g) );//source-sdk-2013

			if( CUBEMAP && CSGO )
				envMapColor = fSpecExpMap * mix( envMapColor, envMapColor * baseColor.rgb * g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost.w, vec3(vSpecExpMap.g) );
		}
	}
	else vSpecularTint = g_SpecularBoost * g_SpecularTint ;

	vec3 albedo = baseColor.rgb;
	vec3 specularLighting = vec3( 0.0f, 0.0f, 0.0f );
	vec3 rimLighting = vec3( 0.0f, 0.0f, 0.0f );
    vec3 specularLightingFromPhong;

    PixelShaderDoSpecularLighting( vWorldPos, worldSpaceNormal,
        fSpecExp, vEyeDir, vLightAtten,
        NUM_LIGHTS, lightinfo, PHONGWARPTEXTURE, SpecularWarpSampler, fFresnelRanges, RIMLIGHT, g_RimExponent,
        flDirectionalShadow,
        // Outputs
        specularLightingFromPhong, rimLighting );

	specularLighting += (specularLightingFromPhong);

    // Modulate with spec mask and tint (modulated by boost above)
	specularLighting *= (fSpecMask * vSpecularTint);
	// If we didn't already apply Fresnel to specular warp, modulate the specular
	if ( !PHONGWARPTEXTURE )
	{
		specularLighting *= fFresnelRanges;
	}

	if ( TINTMASKTEXTURE )
    {
		// Optionally use inverseblendtint texture to blend in the diffuse modulation (saturated add of g_fInverseBlendTintByBaseAlpha turns this on/off)
		float tintMask = sampleWithDefault(ColorTintSampler, inputs.sparse_coord, 1.0);
		diffuseLighting *= mix( vec3(1.0f), g_DiffuseModulation.rgb, saturate( tintMask + g_fInverseBlendTintByBaseAlpha ) );
    }
	else
		// Optionally use basealpha to blend in the diffuse modulation (saturated add of g_fInverseBlendTintByBaseAlpha turns this on/off)
		diffuseLighting *= mix( vec3(1.0f), g_DiffuseModulation.rgb, saturate( baseColor.a + g_fInverseBlendTintByBaseAlpha ) );

	vec3 diffuseComponent = albedo * diffuseLighting;

	if ( SELFILLUM )
	{
		float g_SelfIllumMaskControl = 1.0;
		if ( SELFILLUMFRESNEL == 1 )
		{
			g_SelfIllumMaskControl = 0.0;
			// This will apply a Fresnel term based on the vertex normal (not the per-pixel normal!) to help fake and internal glow look
			vec3 vVertexNormal = vectors.vertexNormal;
			vec3 vSelfIllumMask = getBaseColor( SelfIllumMaskSampler, inputs.sparse_coord ).rgb;
			vSelfIllumMask = mix( baseColor.aaa, vSelfIllumMask, vec3(g_SelfIllumMaskControl) );
			float flSelfIllumFresnel = ( pow( saturate( dot( vVertexNormal.xyz, vEyeDir.xyz ) ), g_SelfIllumScaleBiasExpBrightness.z ) * g_SelfIllumScaleBiasExpBrightness.x ) + g_SelfIllumScaleBiasExpBrightness.y;
			diffuseComponent = mix( diffuseComponent, g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost.rgb * albedo * g_SelfIllumScaleBiasExpBrightness.w, vSelfIllumMask.rgb * saturate( flSelfIllumFresnel ) );

		}
		else
		{
			vec3 vSelfIllumMask = getBaseColor( SelfIllumMaskSampler, inputs.sparse_coord ).rgb;
			vSelfIllumMask = mix( baseColor.aaa, vSelfIllumMask, vec3(g_SelfIllumMaskControl) );
			diffuseComponent = mix( diffuseComponent, g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost.rgb * albedo, vSelfIllumMask );
		}

		diffuseComponent = max( vec3(0.0f), diffuseComponent );
	}

	if ( DETAILTEXTURE )
	{
		diffuseComponent =  TextureCombinePostLighting( diffuseComponent, detailColor, DETAIL_BLEND_MODE, g_SelfIllumTint_and_DetailBlendFactorOrPhongAlbedoBoost.w );
	}


	//render to gamma space ?

	if ( RIMLIGHT )
	{
		float f_fixRim2D = is_perspective?1.0f:0.0f; //block 2DView rimlight
		float fRimFresnel = Fresnel4( worldSpaceNormal, vEyeDir );

		// Add in rim light modulated with tint, mask and traditional Fresnel (not using Fresnel ranges)
		float fRimMultiply = fRimMask * fRimFresnel;
		rimLighting *= fRimMultiply * f_fixRim2D ;

		// Fold rim lighting into specular term by using the max so that we don't really add light twice...
		specularLighting = max( specularLighting, rimLighting );

		// Add in view-ray lookup from ambient cube
		vec3 rimAmbinet = (bAmbient && !bIBL)?ambient_color:cAmbientCube;

		if (CSGO)
			specularLighting += fRimFresnel * fRimMask * g_fRimBoost * rimAmbinet * saturate(dot(worldSpaceNormal, vec3(0, 1, 0)) ) * f_fixRim2D;
		else
			specularLighting += (rimAmbinet * g_fRimBoost) * saturate(fRimMultiply * worldSpaceNormal.y) * f_fixRim2D;//skd-2013
	
		//in fact, they look like same really!
	}

    vec3 result = envMapColor + diffuseComponent;
	if (PHONG) 
		result += specularLighting;

	diffuseShadingOutput(bHDR?ACESFitted(result):result);
	//diffuseShadingOutput(result);
	if ( bHasAlpha && fAlpha < 1.0 )
    {
		alphaKill(inputs.sparse_coord);
	}

}

#include "baked_model.hpp"

#include <cstdio>
#include <cstring>

#include "../labutils/error.hpp"
#include "../labutils/to_string.hpp"
#include "../labutils/vkutil.hpp"
namespace lut = labutils;

namespace
{
	// See cw2-bake/main.cpp for more info
	constexpr char kFileMagic[16] = "\0\0COMP5822Mmesh";
	constexpr char kFileVariant[16] = "default";

	constexpr std::uint32_t kMaxString = 32*1024;

	// functions
	BakedModel load_baked_model_( FILE*, char const* );
}

BakedModel load_baked_model( char const* aModelPath )
{
	FILE* fin = std::fopen( aModelPath, "rb" );
	if( !fin )
		throw lut::Error( "load_baked_model(): unable to open '%s' for reading", aModelPath );

	try
	{
		auto ret = load_baked_model_( fin, aModelPath );
		std::fclose( fin );
		return ret;
	}
	catch( ... )
	{
		std::fclose( fin );
		throw;
	}
}

namespace
{
	void checked_read_( FILE* aFin, std::size_t aBytes, void* aBuffer )
	{
		auto ret = std::fread( aBuffer, 1, aBytes, aFin );

		if( aBytes != ret )
			throw lut::Error( "checked_read_(): expected %zu bytes, got %zu", aBytes, ret );
	}

	std::uint32_t read_uint32_( FILE* aFin )
	{
		std::uint32_t ret;
		checked_read_( aFin, sizeof(std::uint32_t), &ret );
		return ret;
	}
	std::string read_string_( FILE* aFin )
	{
		auto const length = read_uint32_( aFin );

		if( length >= kMaxString )
			throw lut::Error( "read_string_(): unexpectedly long string (%u bytes)", length );

		std::string ret;
		ret.resize( length );

		checked_read_( aFin, length, ret.data() );
		return ret;
	}

	BakedModel load_baked_model_( FILE* aFin, char const* aInputName )
	{
		BakedModel ret;

		// Figure out base path
		char const* pathBeg = aInputName;
		char const* pathEnd = std::strrchr( pathBeg, '/' );
	
		std::string const prefix = pathEnd
			? std::string( pathBeg, pathEnd+1 )
			: ""
		;

		// Read header and verify file magic and variant
		char magic[16];
		checked_read_( aFin, 16, magic );

		if( 0 != std::memcmp( magic, kFileMagic, 16 ) )
			throw lut::Error( "load_baked_model_(): %s: invalid file signature!", aInputName );

		char variant[16];
		checked_read_( aFin, 16, variant );

		if( 0 != std::memcmp( variant, kFileVariant, 16 ) )
			throw lut::Error( "load_baked_model_(): %s: file variant is '%s', expected '%s'", aInputName, variant, kFileVariant );

		// Read texture info
		auto const textureCount = read_uint32_( aFin );
		for( std::uint32_t i = 0; i < textureCount; ++i )
		{
			BakedTextureInfo info;
			info.path = prefix + read_string_( aFin );

			std::uint8_t channels;
			checked_read_( aFin, sizeof(std::uint8_t), &channels );
			info.channels = channels;

			ret.textures.emplace_back( std::move(info) );
		}

		// Read material info
		auto const materialCount = read_uint32_( aFin );
		for( std::uint32_t i = 0; i < materialCount; ++i )
		{
			BakedMaterialInfo info;
			info.baseColorTextureId = read_uint32_( aFin );
			info.roughnessTextureId = read_uint32_( aFin );
			info.metalnessTextureId = read_uint32_( aFin );
			info.alphaMaskTextureId = read_uint32_( aFin );
			info.normalMapTextureId = read_uint32_( aFin );

			assert( info.baseColorTextureId < ret.textures.size() );
			assert( info.roughnessTextureId < ret.textures.size() );
			assert( info.metalnessTextureId < ret.textures.size() );

			ret.materials.emplace_back( std::move(info) );
		}

		// Read mesh data
		auto const meshCount = read_uint32_( aFin );
		for( std::uint32_t i = 0; i < meshCount; ++i )
		{
			BakedMeshData data;
			data.materialId = read_uint32_( aFin );
			assert( data.materialId < ret.materials.size() );

			auto const V = read_uint32_( aFin );
			auto const I = read_uint32_( aFin );

			data.positions.resize( V );
			checked_read_( aFin, V*sizeof(glm::vec3), data.positions.data() );

			data.normals.resize( V );
			checked_read_( aFin, V*sizeof(glm::vec3), data.normals.data() );

			data.texcoords.resize( V );
			checked_read_( aFin, V*sizeof(glm::vec2), data.texcoords.data() );

			data.indices.resize( I );
			checked_read_( aFin, I*sizeof(std::uint32_t), data.indices.data() );

			ret.meshes.emplace_back( std::move(data) );
		}

		// Check
		char byte;
		auto const check = std::fread( &byte, 1, 1, aFin );
		
		if( 0 != check )
			std::fprintf( stderr, "Note: '%s' contains trailing bytes\n", aInputName );

		return ret;
	}
}


void create_mesh(BakedModel aModel, lut::Allocator const& allocator, labutils::VulkanContext const& aContext, std::vector<SceneMesh>& sceneMeshes)
{

	for (unsigned int i = 0; i < aModel.meshes.size(); i++)
	{
		// Creating position, normal and texture buffers
		lut::Buffer vertexPosGPU = lut::create_buffer(
			allocator,
			//sizeof(model.dataTextured.positions),
			sizeof(glm::vec3) * aModel.meshes[i].positions.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexNormGPU = lut::create_buffer(
			allocator,
			sizeof(glm::vec3) * aModel.meshes[i].normals.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexUvGPU = lut::create_buffer(
			allocator,
			sizeof(glm::vec2) * aModel.meshes[i].texcoords.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexIndGPU = lut::create_buffer(
			allocator,
			sizeof(std::uint32_t) * aModel.meshes[i].indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		//Staging Buffers
		lut::Buffer posStaging = lut::create_buffer(
			allocator,
			sizeof(glm::vec3) * aModel.meshes[i].positions.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer normStaging = lut::create_buffer(
			allocator,
			sizeof(glm::vec3) * aModel.meshes[i].normals.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer uvStaging = lut::create_buffer(
			allocator,
			sizeof(glm::vec2) * aModel.meshes[i].texcoords.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer indStaging = lut::create_buffer(
			allocator,
			sizeof(std::uint32_t) * aModel.meshes[i].indices.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(allocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}
		std::memcpy(posPtr, aModel.meshes[i].positions.data(), sizeof(glm::vec3) * aModel.meshes[i].positions.size());
		vmaUnmapMemory(allocator.allocator, posStaging.allocation);

		void* normPtr = nullptr;
		if (auto const res = vmaMapMemory(allocator.allocator, normStaging.allocation, &normPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}
		std::memcpy(normPtr, aModel.meshes[i].normals.data(), sizeof(glm::vec3) * aModel.meshes[i].normals.size());
		vmaUnmapMemory(allocator.allocator, normStaging.allocation);

		void* uvPtr = nullptr;
		if (auto const res = vmaMapMemory(allocator.allocator, uvStaging.allocation, &uvPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());

		}
		std::memcpy(uvPtr, aModel.meshes[i].texcoords.data(), sizeof(glm::vec2) * aModel.meshes[i].texcoords.size());
		vmaUnmapMemory(allocator.allocator, uvStaging.allocation);

		void* indPtr = nullptr;
		if (auto const res = vmaMapMemory(allocator.allocator, indStaging.allocation, &indPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}
		std::memcpy(indPtr, aModel.meshes[i].indices.data(), sizeof(std::uint32_t) * aModel.meshes[i].indices.size());
		vmaUnmapMemory(allocator.allocator, indStaging.allocation);


		// We need to ensure that the Vulkan resources are alive until all the 
		// transfers have completed. For simplicity, we will just wait for the 
		// operations to complete with a fence.
		lut::Fence uploadComplete = lut::create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers 
		// This uses a separate command pool for simplicity. 
		lut::CommandPool uploadPool = create_command_pool(aContext);
		VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy pcopy{};
		pcopy.size = sizeof(glm::vec3) * aModel.meshes[i].positions.size();

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ncopy{};
		ncopy.size = sizeof(glm::vec3) * aModel.meshes[i].normals.size();

		vkCmdCopyBuffer(uploadCmd, normStaging.buffer, vertexNormGPU.buffer, 1, &ncopy);

		lut::buffer_barrier(uploadCmd,
			vertexNormGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);
		
		VkBufferCopy tcopy{};
		tcopy.size = sizeof(glm::vec2) * aModel.meshes[i].texcoords.size();
		
		vkCmdCopyBuffer(uploadCmd, uvStaging.buffer, vertexUvGPU.buffer, 1, &tcopy);
		
		lut::buffer_barrier(uploadCmd,
			vertexUvGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);
		
		VkBufferCopy icopy{};
		ncopy.size = sizeof(std::uint32_t) * aModel.meshes[i].indices.size();

		vkCmdCopyBuffer(uploadCmd, indStaging.buffer, vertexIndGPU.buffer, 1, &ncopy);

		lut::buffer_barrier(uploadCmd,
			vertexIndGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);
		
		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
		
		// Submit transfer commands 
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}

		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}


		
		sceneMeshes.emplace_back(SceneMesh{
			std::move(vertexPosGPU),
			std::move(vertexNormGPU),
			std::move(vertexUvGPU),
			std::move(vertexIndGPU),
			unsigned int(aModel.meshes[i].indices.size())// IndexCount
			});
	}
	
}
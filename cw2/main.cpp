#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>
#include <unordered_map>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <volk/volk.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"


namespace
{
	struct ShaderPath {
		char const* kVertShaderPath;
		char const* kFragShaderPath;
	};

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
#		define SHADERDIR_ "assets/cw2/shaders/"

		ShaderPath defaultShaderPath{ SHADERDIR_ "default.vert.spv" , SHADERDIR_ "default.frag.spv" };
		//ShaderPath untexturedShaderPath{ SHADERDIR_ "untextured.vert.spv", SHADERDIR_ "untextured.frag.spv" };

#		undef SHADERDIR_

			// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		// Camera settings. 
		// These are determined empirically 
		constexpr float kCameraBaseSpeed = 1.7f; // units/second 
		constexpr float kCameraFastMult = 5.f; // speed multiplier 
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier 

		constexpr float kCameraMouseSensitivity = 0.01f; // radians per pixel
	}
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	// Local types/structures:
	namespace glsl
	{
		struct SceneUniform
		{ // Note: need to be careful about the packing/alignment here! 
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;

			glm::vec3 cameraPosition;
			alignas(16) glm::vec3 lightPosition{ 0.0f, 1.0f, 0.0f };
			alignas(16) glm::vec4 lightColor{ 1.0f, 1.0f, 1.0f, 1.0f };
		};
	}
	enum class EInputState {
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState {
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};
	// Local functions:
	lut::RenderPass create_render_pass(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&, VkDescriptorType);
	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, std::vector<VkDescriptorSetLayout>, unsigned int pushConstantSize = 0);
	lut::Pipeline create_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, ShaderPath);
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator);

	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	void update_user_state(UserState&, float aElapsedTime);

	void create_swapchain_framebuffers(lut::VulkanWindow const&, VkRenderPass, std::vector<lut::Framebuffer>&, VkImageView aDepthView);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const& aState
	);

	void record_commands(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer,
		VkPipelineLayout aPipeLayout, VkPipeline aPipe, VkExtent2D const& aImageExtent, std::vector<SceneMesh> const&,
		VkBuffer aSceneUBO, glsl::SceneUniform const& aSceneUniform,
		VkDescriptorSet aSceneDescriptors,
		std::vector <VkDescriptorSet> aTexDescriptors, BakedModel* aModel, std::unordered_map <unsigned int, std::vector<unsigned int>> MaterialMeshesMap);


	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
}

int main() try
{
	//TODO-implement me.
	// Create our Vulkan Window
	lut::VulkanWindow window = lut::make_vulkan_window();

	// Configure the GLFW window
	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	
	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	//Creaing resourses for rendering
	lut::RenderPass renderPass = create_render_pass(window);

	//create scene descriptor set layout
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);

	//TODO- (Section 4) create object descriptor set layout
	lut::DescriptorSetLayout texturedobjectLayout = create_object_descriptor_layout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, std::vector< VkDescriptorSetLayout> {sceneLayout.handle, texturedobjectLayout.handle});
	lut::Pipeline pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle, cfg::defaultShaderPath);

	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);
	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	//////////////////////////////////////////////////////////////////////////////////

	BakedModel bakedModel = load_baked_model("assets\\cw2\\sponza-pbr.comp5822mesh");
	printf("\n Number of Meshes = %d \n", bakedModel.meshes.size());
	
	
	//Create a buffer for each mesh
	std::vector<SceneMesh> sceneMeshes;
	sceneMeshes.reserve(bakedModel.meshes.size());
	create_mesh(bakedModel, allocator, window, sceneMeshes);
	
	// Create a Material->Meshes Map so that all meshes with same material can be drawn consequently
	// thus redusing the number of descriptor binding
	std::unordered_map <unsigned int, std::vector<unsigned int>> MaterialMeshesMap;
	for (unsigned int i = 0; i < bakedModel.meshes.size(); i++)
	{
		MaterialMeshesMap[bakedModel.meshes[i].materialId].emplace_back(i);
	}
	
	//create scene uniform buffer with lut::create_buffer()
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);
	
	
	//create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);
	// allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	//initialize descriptor set with vkUpdateDescriptorSets
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}


	//load textures into image
	// For each texture in model, create image 
	std::vector <lut::Image> texImages;
	texImages.reserve(bakedModel.textures.size());
	for (unsigned int i = 0; i < bakedModel.textures.size(); i++)
	{	
		{
			lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
			//Create Image for the texture
			texImages.push_back ( lut::load_image_texture2d(bakedModel.textures[i].path.c_str(), window, loadCmdPool.handle, allocator));
			// Create view for the texture
			//texImageViews.emplace_back(lut::create_image_view_texture2d(window, texImages.image, VK_FORMAT_R8G8B8A8_SRGB));
		}	
	}

	std::vector <lut::ImageView> texImageViews;
	texImageViews.reserve(texImages.size());
	for (unsigned int i = 0; i < texImages.size(); i++)
	{
		// Create view for the texture
		texImageViews.emplace_back(lut::create_image_view_texture2d(window, texImages[i].image, VK_FORMAT_R8G8B8A8_SRGB));
	}

	// create default texture sampler
	lut::Sampler defaultSampler = lut::create_default_sampler(window);

	// allocate and initialize descriptor sets for texture
	std::vector <VkDescriptorSet> texDescriptors;
	for (unsigned int i = 0; i < texImageViews.size(); i++)
	{
		VkDescriptorSet oDescriptors = lut::alloc_desc_set(window, dpool.handle, texturedobjectLayout.handle);
		{
			VkWriteDescriptorSet desc[1]{};

			VkDescriptorImageInfo textureInfo{};
			textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo.imageView = texImageViews[i].handle;
			textureInfo.sampler = defaultSampler.handle;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = oDescriptors;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		}
		texDescriptors.push_back(oDescriptors);
	}




	// Application main loop
	bool recreateSwapchain = false;
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//TODO: re-create swapchain and associated resources!
			// We need to destroy several objects, which may still be in use by 
			// the GPU. Therefore, first wait for the GPU to finish processing. 
			vkDeviceWaitIdle(window.device);

			// Recreate them 
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			if (changes.changedSize)
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			if (changes.changedSize)
				pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle, cfg::defaultShaderPath);

			recreateSwapchain = false;
			continue;
		}

		// Acquire next swap chain image 
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}


		//wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());
		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		// Update state 
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);

		//Update uniforms
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);

		//Record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipeLayout.handle,
			pipe.handle,
			window.swapchainExtent,
			sceneMeshes,
			sceneUBO.buffer,
			sceneUniforms,
			sceneDescriptors,
			texDescriptors,
			&bakedModel,
			MaterialMeshesMap
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		// Present the results 
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinished.handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			recreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", imageIndex, lut::to_string(presentRes).c_str());
		}

	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);
	///

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		// Note: the stencilLoadOp & stencilStoreOp members are left initialized 
		// to 0 (=DONT CARE). The image format (R8G8B8A8 SRGB) of the color 
		// attachment does not have a stencil component, so these are ignored 
		// either way. 

		//ATTACHMENT
		//framebuffer
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling 
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		//depthbuffer
		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//SUBPASS
		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // the zero refers to attachments[0] declared earlier. 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; // this refers to attachments[1] 
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		//no explicit subpass dependencies 

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0; //changed!
		passInfo.pDependencies = nullptr; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow, VkDescriptorType aDescriptorType)
	{
		//throw lut::Error("Not yet implemented"); //TODO: (Section 4) implement me!
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // this must match the shaders
		bindings[0].descriptorType = aDescriptorType;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext,
		std::vector<VkDescriptorSetLayout> aLayout, unsigned int apushConstantSize)
	{
		//throw lut::Error("Not yet implemented"); //TODO: implement me!
		/*VkDescriptorSetLayout layouts[] = { aLayout[0] };
		for (int i = 1; i < aLayout.size(); i++) {
			layouts[i] = aLayout[i];
		}*/
		//VkDescriptorSetLayout layouts[] = { 
			// Order must match the set = N in the shaders 
		//	aSceneLayout, // set 0 
		//	aObjectLayout
		//};

		//VkPushConstantRange colorPushConstant;
		//if (apushConstantSize != 0)
		//{

		//	//this push constant range starts at the beginning
		//	colorPushConstant.offset = 0;
		//	//this push constant range takes up the size of a MeshPushConstants struct
		//	colorPushConstant.size = apushConstantSize;
		//	//this push constant range is accessible only in the vertex shader
		//	//colorPushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		//	colorPushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		//}

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = aLayout.size();
		layoutInfo.pSetLayouts = aLayout.data();
		/*if (apushConstantSize != 0)
		{
			layoutInfo.pushConstantRangeCount = 1;
			layoutInfo.pPushConstantRanges = &colorPushConstant;
		}
		else
		{*/
			layoutInfo.pushConstantRangeCount = 0;
			layoutInfo.pPushConstantRanges = nullptr;
		//}


		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}


	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, ShaderPath aShaderPath)
	{
		//throw lut::Error("Not yet implemented"); //TODO: implement me!
		lut::ShaderModule vert = lut::load_shader_module(aWindow, aShaderPath.kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, aShaderPath.kFragShaderPath);

		// Define shader stages in the pipeline 
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		/// //////////////////////

		VkVertexInputBindingDescription vertexInputs[3]{};
		//position buffer
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(glm::vec3);
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		//normals buffer
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(glm::vec3);
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		//uv buffer	
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(glm::vec2);
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[2].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		// Define which primitive (point, line, triangle, ...) the input is 
		// assembled into for rasterization. 
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions 
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = aWindow.swapchainExtent.width;
		viewport.height = aWindow.swapchainExtent.height;
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options 
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required.

		// Define multisampling state 
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we don�t do any
		// blending, so we can ignore most of the members. 
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Create pipeline 
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1,
			&pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass,
		std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		//throw lut::Error("Not yet implemented"); //TODO: implement me!
		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView };

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer 
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;

			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}


	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		//throw lut::Error("Not yet implemented"); //TODO- (Section 6) implement me!
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo,
			&image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		// Create the image view 
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
		VK_IMAGE_ASPECT_DEPTH_BIT,
		0, 1,
		0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return{ std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}


	void record_commands(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipelineLayout aPipeLayout, 
		VkPipeline aPipe,  VkExtent2D const& aImageExtent, std::vector<SceneMesh> const& sceneMeshes, VkBuffer aSceneUBO,
		glsl::SceneUniform const& aSceneUniform, VkDescriptorSet aSceneDescriptors,
		std::vector <VkDescriptorSet> aTexDescriptors, BakedModel* aModel, std::unordered_map <unsigned int, std::vector<unsigned int>> aMaterialMeshesMap)
	{
		//throw lut::Error("Not yet implemented"); //TODO: implement me!
		// Begin recording commands 
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Upload scene uniforms 
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		// Clear values
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background. 
		clearValues[0].color.float32[1] = 0.1f; // If we were debugging, this would potentially 
		clearValues[0].color.float32[2] = 0.1f; // help us see whether the render pass took 
		clearValues[0].color.float32[3] = 1.f; // place, even if nothing else was drawn. 

		clearValues[1].depthStencil.depth = 1.f;

		// Begin render pass 
		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);


		// Begin drawing with graphics pipeline 
		//
		
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
			0, 1, &aSceneDescriptors, 0, nullptr);

	
		for (unsigned int matID = 0; matID < aModel->materials.size(); matID++)
		{
			// Bind Material descriptors
			// Base Color Texture
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
				1, 1, &aTexDescriptors[
					aModel->materials[
						matID
					].baseColorTextureId
				], 0, nullptr);

			// Roughness Texture
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
				1, 1, &aTexDescriptors[
					aModel->materials[
						matID
					].roughnessTextureId
				], 0, nullptr);

			// Metalness Texture
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
				1, 1, &aTexDescriptors[
					aModel->materials[
						matID
					].metalnessTextureId
				], 0, nullptr);

			// AlphaMask Texture
			//vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
			//	1, 1, &aTexDescriptors[
			//		aModel->materials[
			//			matID
			//		].alphaMaskTextureId
			//	], 0, nullptr);
			//// NormalMap Texture
			//vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
			//	1, 1, &aTexDescriptors[
			//		aModel->materials[
			//			matID
			//		].normalMapTextureId
			//	], 0, nullptr);

			for (unsigned int i = 0; i < aMaterialMeshesMap[matID].size(); i++)
			{
				// Bind vertex input 
				VkBuffer vBuffers[3] = { sceneMeshes[aMaterialMeshesMap[matID][i]].positions.buffer,
										sceneMeshes[aMaterialMeshesMap[matID][i]].normals.buffer,
										sceneMeshes[aMaterialMeshesMap[matID][i]].texcoords.buffer };
				VkDeviceSize offsets[3]{};
				vkCmdBindVertexBuffers(aCmdBuff, 0, 3, vBuffers, offsets);

				//Bind Index Buffer
				vkCmdBindIndexBuffer(aCmdBuff, sceneMeshes[aMaterialMeshesMap[matID][i]].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				//Draw
				vkCmdDrawIndexed(aCmdBuff, sceneMeshes[aMaterialMeshesMap[matID][i]].indexCount, 1, 0, 0, 0);
			}

		}
		

		//For each mesh, Bind buffers and draw
		//for (unsigned int i = 0; i < sceneMeshes.size(); i++)
		//{
		//	vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeLayout,
		//		1, 1, &aTexDescriptors[
		//				aModel->materials[
		//				aModel->meshes[i].materialId
		//			].baseColorTextureId
		//		], 0, nullptr);

		////	// Bind vertex input 
		//	VkBuffer vBuffers[3] = { sceneMeshes[i].positions.buffer, 
		//							sceneMeshes[i].normals.buffer, 
		//							sceneMeshes[i].texcoords.buffer };
		//	VkDeviceSize offsets[3]{};
		//	vkCmdBindVertexBuffers(aCmdBuff, 0, 3, vBuffers, offsets);

		////	//Bind Index Buffer
		//	vkCmdBindIndexBuffer(aCmdBuff, sceneMeshes[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		//	
		//	//Draw
		//	vkCmdDrawIndexed(aCmdBuff, sceneMeshes[i].indexCount, 1, 0, 0, 0);

		//}
		
		// End the render pass 
		vkCmdEndRenderPass(aCmdBuff);


		// End command recording 
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}


	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		//throw lut::Error("Not yet implemented"); //TODO: implement me!
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

}


namespace 
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			// Only update the rotation on the second frame of mouse 
			// navigation. This ensures that the previousX and Y variables are 
			// initialized to sensible values.
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));
	}



	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aState)
	{
		//TODO- (Section 3) initialize SceneUniform members
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);
		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis 	
		aSceneUniforms.camera = glm::inverse(aState.camera2world);
		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
		aSceneUniforms.cameraPosition = aState.camera2world[3];
	}
}


//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
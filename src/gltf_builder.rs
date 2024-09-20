use std::{
    borrow::Cow,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};

use gltf::{
    binary::Header,
    buffer::Target,
    json::{
        self,
        buffer::{Stride, View},
        mesh::Primitive,
        root::Get,
        texture,
        validation::{Checked, USize64},
        Accessor, Animation, Buffer, Camera, Extras, Image, Index, Material, Mesh, Node, Root,
        Scene, Skin, Texture,
    },
    Glb,
};

#[derive(Debug, Clone, Default)]
pub struct GltfBuilder {
    root: Root,
    blobs: Vec<Vec<u8>>,
}

impl GltfBuilder {
    /// Create a new gltf builder in binary mode
    pub fn new() -> Self {
        Default::default()
    }

    #[track_caller]
    /// Push a gltf element to the builder
    pub fn push<T>(&mut self, value: T) -> Index<T>
    where
        Self: AsMut<Vec<T>>,
    {
        Index::push(self.as_mut(), value)
    }

    pub fn push_buffer<T>(
        &mut self,
        name: Option<String>,
        buffer: Vec<T>,
        uri: Option<String>,
    ) -> Index<Buffer> {
        let byte_buffer = vec_to_u8_vec(buffer);
        let index = self.root.push(Buffer {
            byte_length: USize64::from(byte_buffer.len()),
            name,
            uri,
            extensions: None,
            extras: Extras::default(),
        });
        self.blobs.push(byte_buffer);
        index
    }

    pub fn push_buffer_with_view<T>(
        &mut self,
        name: Option<String>,
        buffer: Vec<T>,
        stride: Option<usize>,
        uri: Option<String>,
    ) -> Index<View> {
        let t_size = core::mem::size_of::<T>();
        let buffer_length = buffer.len() * t_size;
        let buffer = self.push_buffer(None, buffer, uri);
        let byte_stride = Some(Stride(stride.unwrap_or(1) * t_size));
        self.push_view(View {
            buffer,
            byte_length: USize64::from(buffer_length),
            byte_offset: None,
            byte_stride,
            extensions: Default::default(),
            extras: Default::default(),
            name,
            target: Some(Checked::Valid(Target::ArrayBuffer)),
        })
    }

    // fn get_buffer_offset(&self, buffer: Index<Buffer>) -> u64 {
    //     self.blobs[..buffer.value()]
    //         .iter()
    //         .map(|it| it.len())
    //         .sum::<usize>() as u64
    // }

    pub fn push_view(&mut self, view: View) -> Index<View> {
        // let mut view = view;
        // let buffer = view.buffer;
        // let original_offset = view.byte_offset.unwrap_or_default().0;
        // view.buffer = Index::new(0);
        // view.byte_offset = Some((self.get_buffer_offset(buffer) + original_offset).into());
        self.root.push(view)
    }

    pub fn push_accessor_vec3(
        &mut self,
        name: Option<String>,
        buffer_view: Index<View>,
        offset: usize,
        count: usize,
        min: Option<[f32; 3]>,
        max: Option<[f32; 3]>,
    ) -> Index<Accessor> {
        let t_size = core::mem::size_of::<f32>();
        self.push(json::Accessor {
            buffer_view: Some(buffer_view),
            byte_offset: Some(USize64::from(offset * t_size)),
            count: USize64::from(count),
            component_type: Checked::Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::F32,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Checked::Valid(json::accessor::Type::Vec3),
            min: min.map(|min| json::Value::from(Vec::from(min))),
            max: max.map(|max| json::Value::from(Vec::from(max))),
            name,
            normalized: false,
            sparse: None,
        })
    }

    #[allow(dead_code)]
    pub fn push_accessor_vec3_u32(
        &mut self,
        name: Option<String>,
        buffer_view: Index<View>,
        offset: usize,
        count: usize,
    ) -> Index<Accessor> {
        let t_size = core::mem::size_of::<u32>();
        self.push(json::Accessor {
            buffer_view: Some(buffer_view),
            byte_offset: Some(USize64::from(offset * t_size)),
            count: USize64::from(count),
            component_type: Checked::Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::U32,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Checked::Valid(json::accessor::Type::Vec3),
            min: None,
            max: None,
            name,
            normalized: false,
            sparse: None,
        })
    }

    #[allow(dead_code)]
    pub fn push_accessor_u32(
        &mut self,
        name: Option<String>,
        buffer_view: Index<View>,
        offset: usize,
        count: usize,
    ) -> Index<Accessor> {
        let t_size = core::mem::size_of::<u32>();
        self.push(json::Accessor {
            buffer_view: Some(buffer_view),
            byte_offset: Some(USize64::from(offset * t_size)),
            count: USize64::from(count),
            component_type: Checked::Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::U32,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Checked::Valid(json::accessor::Type::Scalar),
            min: None,
            max: None,
            name,
            normalized: false,
            sparse: None,
        })
    }

    pub fn push_mesh(
        &mut self,
        name: Option<String>,
        primitives: Vec<Primitive>,
        weights: Option<Vec<f32>>,
    ) -> Index<Mesh> {
        self.push(json::Mesh {
            extensions: Default::default(),
            extras: Default::default(),
            name,
            primitives,
            weights,
        })
    }

    pub fn push_node(&mut self, mesh: Index<Mesh>) -> Index<Node> {
        self.push(json::Node {
            mesh: Some(mesh),
            ..Default::default()
        })
    }

    pub fn push_scene(&mut self, nodes: Vec<Index<Node>>) -> Index<Scene> {
        self.push(json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            nodes,
        })
    }

    pub fn set_default_scene(&mut self, scene: Option<Index<Scene>>) {
        self.root.scene = scene;
    }

    fn compute_glb_len(&self, json_data_size: usize) -> usize {
        // NOTE: glb chunks must be 4-bytes aligned (padded with 0s at the end)
        let chunk_header_size = 8; // chunk length (u32) + chunk type (u32)
        let bin_chunk_size = chunk_header_size
            + align_to_multiple_of_four(self.blobs.iter().map(|it| it.len()).sum::<usize>());
        let json_chunk_size = align_to_multiple_of_four(json_data_size) + chunk_header_size;
        let glb_header_size = 12; // magic (u32) + version (u32) + file length (u32)
        glb_header_size + json_chunk_size + bin_chunk_size
    }

    fn combine_bin_chunk(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for blob in &self.blobs {
            result.extend_from_slice(blob);
        }
        result
    }

    #[allow(dead_code)]
    pub fn to_json(&self) -> String {
        json::serialize::to_string(&self.root).expect("Serialization error")
    }

    /// Set the URI of all buffers to `"{prefix}_{index}.bin"`
    #[allow(dead_code)]
    pub fn set_buffers_uri(&mut self, prefix: &str) {
        for (i, buffer) in self.root.buffers.iter_mut().enumerate() {
            buffer.uri = Some(format!("{}_{}.bin", prefix, i));
        }
    }

    /// Set the URI of a buffer to `uri`
    pub fn set_buffer_uri(&mut self, index: usize, uri: Option<String>) -> Result<(), String> {
        let buffer = self
            .root
            .buffers
            .get_mut(index)
            .ok_or_else(|| "Unable to find buffer".to_string())?;
        buffer.uri = uri;
        Ok(())
    }

    pub fn write_all_buffers(&self, dir: impl AsRef<Path>) -> Result<(), String> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err("Invalid output directory.".to_string());
        }

        for (i, buffer) in self.root.buffers.iter().enumerate() {
            if let Some(uri) = &buffer.uri {
                if i >= self.blobs.len() {
                    return Err(format!("Failed to get content of buffer {}", i));
                }
                let mut path = dir.to_path_buf();
                path.push(uri);
                println!(
                    "Write: {} ({}KB)",
                    path.display(),
                    self.blobs[i].len() / 1024
                );
                let f = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(path)
                    .expect("Unable to open file");
                let mut bufw = BufWriter::new(f);
                bufw.write_all(self.blobs[i].as_slice())
                    .expect("Unable to write to file");
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn get_buffer_offset(&self, buffer: Index<Buffer>) -> u64 {
        self.blobs[..buffer.value()]
            .iter()
            .map(|it| it.len())
            .sum::<usize>() as u64
    }

    fn compute_buffers_offsets(&self) -> Vec<usize> {
        let mut res = Vec::new();
        let mut accum = 0;
        for b in self.blobs.iter() {
            res.push(accum);
            accum += b.len();
        }
        res
    }

    fn compute_buffers_len(&self) -> usize {
        self.blobs.iter().map(|it| it.len()).sum::<usize>()
    }

    pub fn merge_gltf_buffers(&self) -> Result<GltfBuilder, String> {
        let root = &self.root;
        let blobs = &self.blobs;
        debug_assert!(blobs.len() == root.buffers.len());
        if blobs.len() != root.buffers.len() {
            return Err("Invalid input data".to_string());
        }
        for (i, buffer) in root.buffers.iter().enumerate() {
            if buffer.byte_length.0 != blobs[i].len() as u64 {
                return Err("Invalid input data".to_string());
            }
        }

        if root.buffers.len() <= 1 {
            return Ok(self.clone());
        }

        let bin_len = self.compute_buffers_len();
        let offsets = self.compute_buffers_offsets();
        debug_assert!(blobs.len() == offsets.len());

        let mut root = root.clone();

        root.buffers.truncate(1);
        root.buffers[0].byte_length = USize64(bin_len as u64);

        for view in &mut root.buffer_views {
            let index = view.buffer.value();
            view.buffer = Index::new(0);
            view.byte_offset = Some(USize64(offsets[index] as u64));
        }

        Ok(GltfBuilder {
            root,
            blobs: vec![self.combine_bin_chunk()],
        })
    }

    /// @param out_dir: only for text format. The file in which to write the binary data
    pub fn to_glb(&self) -> Result<Glb, String> {
        debug_assert_eq!(self.root.buffers.len(), self.blobs.len());

        let mut builder = self.merge_gltf_buffers().unwrap();
        let bin_chunk = builder.blobs.remove(0);

        let json_string = json::serialize::to_string(&builder.root).expect("Serialization error");
        let glb_length = self.compute_glb_len(json_string.len());

        let header = Header {
            magic: *b"glTF",
            version: 2,
            length: (glb_length)
                .try_into()
                .expect("file size exceeds binary glTF limit"),
        };
        Ok(Glb {
            header,
            json: Cow::Owned(json_string.into_bytes()),
            bin: Some(Cow::Owned(bin_chunk)),
        })
    }

    pub fn write_to_gltf<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        json::serialize::to_writer(writer, &self.root).expect("Serialization error");
        Ok(())
    }
}

fn align_to_multiple_of_four(n: usize) -> usize {
    (n + 3) & !3
}

// fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
// let byte_length = vec.len() * std::mem::size_of::<T>();
// let byte_capacity = vec.capacity() * std::mem::size_of::<T>();
// let alloc = vec.into_boxed_slice();
// let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
// let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
// while new_vec.len() % 4 != 0 {
// new_vec.push(0); // pad to multiple of four bytes
// }
// new_vec
// }

macro_rules! impl_get {
    ($ty:ty, $field:ident) => {
        impl<'a> Get<$ty> for GltfBuilder {
            fn get(&self, index: Index<$ty>) -> Option<&$ty> {
                self.root.$field.get(index.value())
            }
        }
        impl AsRef<[$ty]> for GltfBuilder {
            fn as_ref(&self) -> &[$ty] {
                &self.root.$field
            }
        }
        impl AsMut<Vec<$ty>> for GltfBuilder {
            fn as_mut(&mut self) -> &mut Vec<$ty> {
                &mut self.root.$field
            }
        }
    };
}

pub trait IndexMath {
    fn add(&mut self, value: usize);
}

impl<T> IndexMath for Index<T> {
    fn add(&mut self, value: usize) {
        *self = Index::new((self.value() + value) as u32);
    }
}

#[allow(dead_code)]
fn merge_gltf_roots(a: Root, b: Root) -> Root {
    let mut result = a;
    let mut append = b;

    let orig_accessors_count = result.accessors.len();
    // let orig_animations_count = result.animations.len();
    let orig_buffers_count = result.buffers.len();
    let orig_buffer_views_count = result.buffer_views.len();
    let orig_cameras_count = result.cameras.len();
    let orig_images_count = result.images.len();
    let orig_materials_count = result.materials.len();
    let orig_meshes_count = result.meshes.len();
    let orig_nodes_count = result.nodes.len();
    let orig_samplers_count = result.samplers.len();
    // let orig_scenes_count = result.scenes.len();
    let orig_skins_count = result.skins.len();
    let orig_textures_count = result.textures.len();

    result.samplers.append(&mut append.samplers);
    result.images.append(&mut append.images);
    result.buffers.append(&mut append.buffers);
    result.cameras.append(&mut append.cameras);

    for texture in &mut append.textures {
        if let Some(sampler) = &mut texture.sampler {
            sampler.add(orig_samplers_count);
        }
        texture.source.add(orig_images_count);
    }
    result.textures.append(&mut append.textures);

    for material in &mut append.materials {
        if let Some(tex) = &mut material.normal_texture {
            tex.index.add(orig_textures_count);
        }
        if let Some(tex) = &mut material.emissive_texture {
            tex.index.add(orig_textures_count);
        }
        if let Some(tex) = &mut material.occlusion_texture {
            tex.index.add(orig_textures_count);
        }
        if let Some(tex) = &mut material.pbr_metallic_roughness.base_color_texture {
            tex.index.add(orig_textures_count);
        }
        if let Some(tex) = &mut material.pbr_metallic_roughness.metallic_roughness_texture {
            tex.index.add(orig_textures_count);
        }
        if let Some(ext) = &mut material.extensions {
            #[cfg(feature = "KHR_materials_pbrSpecularGlossiness")]
            if let Some(pbr) = &mut ext.pbr_specular_glossiness {
                if let Some(tex) = &mut pbr.diffuse_texture {
                    tex.index.add(orig_textures_count);
                }
                if let Some(tex) = &mut pbr.specular_glossiness_texture {
                    tex.index.add(orig_textures_count);
                }
            }
        }
    }

    for buffer_view in &mut append.buffer_views {
        buffer_view.buffer.add(orig_buffers_count);
    }
    result.buffer_views.append(&mut append.buffer_views);

    for accessor in &mut append.accessors {
        if let Some(view) = &mut accessor.buffer_view {
            view.add(orig_buffer_views_count);
        }
    }
    result.accessors.append(&mut append.accessors);

    for mesh in &mut append.meshes {
        for primitive in &mut mesh.primitives {
            primitive.attributes.iter_mut().for_each(|(_, attribute)| {
                attribute.add(orig_accessors_count);
            });
            if let Some(indices) = &mut primitive.indices {
                indices.add(orig_accessors_count);
            }
            if let Some(material) = &mut primitive.material {
                material.add(orig_materials_count);
            }
            if let Some(morph_targets) = &mut primitive.targets {
                for morph_target in morph_targets {
                    if let Some(positions) = &mut morph_target.positions {
                        positions.add(orig_accessors_count);
                    }
                    if let Some(normals) = &mut morph_target.normals {
                        normals.add(orig_accessors_count);
                    }
                    if let Some(tangents) = &mut morph_target.tangents {
                        tangents.add(orig_accessors_count);
                    }
                }
            }
        }
    }
    result.meshes.append(&mut append.meshes);

    for node in &mut append.nodes {
        if let Some(camera) = &mut node.camera {
            camera.add(orig_cameras_count);
        }
        if let Some(mesh) = &mut node.mesh {
            mesh.add(orig_meshes_count);
        }
        if let Some(skin) = &mut node.skin {
            skin.add(orig_skins_count);
        }
        if let Some(children) = &mut node.children {
            for child in children {
                child.add(orig_nodes_count);
            }
        }
        #[cfg(feature = "KHR_lights_punctual")]
        if let Some(extensions) = &mut node.extensions {
            if let Some(khr_lights_punctual) = &mut extensions.khr_lights_punctual {
                // TODO: NOT WORKING: where is the list list located ?
                khr_lights_punctual.light.add(orig_lights_count);
            }
        }
    }
    result.nodes.append(&mut append.nodes);

    for animation in &mut append.animations {
        for sampler in &mut animation.samplers {
            sampler.input.add(orig_accessors_count);
            sampler.output.add(orig_accessors_count);
        }
        // NOTE: Do not increment animation.channels[*].sampler.
        // Because it is the index of the sampler inside the animation.
        // So the index doesn't change.
    }
    result.animations.append(&mut append.animations);

    for skin in &mut append.skins {
        // inverse_bind_matrices
        // joints
        // skeleton
        if let Some(bind_matrix) = &mut skin.inverse_bind_matrices {
            bind_matrix.add(orig_accessors_count);
        }
        for joint in &mut skin.joints {
            joint.add(orig_nodes_count);
        }
        if let Some(skeleton) = &mut skin.skeleton {
            skeleton.add(orig_nodes_count);
        }
    }
    result.skins.append(&mut append.skins);

    for scene in &mut append.scenes {
        for node in &mut scene.nodes {
            node.add(orig_nodes_count);
        }
    }
    result.scenes.append(&mut append.scenes);

    debug_assert!(append.accessors.is_empty());
    debug_assert!(append.animations.is_empty());
    debug_assert!(append.buffers.is_empty());
    debug_assert!(append.buffer_views.is_empty());
    debug_assert!(append.cameras.is_empty());
    debug_assert!(append.images.is_empty());
    debug_assert!(append.materials.is_empty());
    debug_assert!(append.meshes.is_empty());
    debug_assert!(append.nodes.is_empty());
    debug_assert!(append.samplers.is_empty());
    debug_assert!(append.scenes.is_empty());
    debug_assert!(append.skins.is_empty());
    debug_assert!(append.textures.is_empty());

    result
}

impl_get!(Accessor, accessors);
impl_get!(Animation, animations);
// impl_get!(Buffer, buffers); // Use push_buffer instead
impl_get!(View, buffer_views);
impl_get!(Camera, cameras);
impl_get!(Image, images);
impl_get!(Material, materials);
impl_get!(Mesh, meshes);
impl_get!(Node, nodes);
impl_get!(texture::Sampler, samplers);
impl_get!(Scene, scenes);
impl_get!(Skin, skins);
impl_get!(Texture, textures);

fn vec_to_u8_vec<T: Sized>(vec: Vec<T>) -> Vec<u8> {
    unsafe {
        let t_size = core::mem::size_of::<T>();
        let result = Vec::from_raw_parts(
            vec.as_ptr() as *mut u8,
            vec.len() * t_size,
            vec.capacity() * t_size,
        );
        std::mem::forget(vec);
        result
    }
}

#[allow(dead_code)]
unsafe fn vec_as_u8_slice<T: Sized>(data: &[T]) -> &[u8] {
    core::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
}

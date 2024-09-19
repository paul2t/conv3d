use std::borrow::Cow;

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
    text_format: bool,
    bin_buffer_name: Option<String>,
    root: Root,
    blob: Vec<Vec<u8>>,
}

impl GltfBuilder {
    /// Create a new gltf builder in binary mode
    pub fn new_glb() -> Self {
        Default::default()
    }

    /// Create a new gltf builder in text mode
    #[allow(dead_code)]
    pub fn new_gltf(bin_buffer_name: String) -> Self {
        Self {
            text_format: true,
            bin_buffer_name: Some(bin_buffer_name),
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn set_binary(&mut self) {
        self.text_format = false;
        for buffer in &mut self.root.buffers {
            buffer.uri = None;
        }
    }

    #[allow(dead_code)]
    pub fn set_text(&mut self, bin_buffer_name: String) {
        self.text_format = true;
        assert_eq!(
            self.root.buffers.len(),
            1,
            "Only one buffer allowed. Use push_buffer to concatenate the buffers into one."
        );
        for buffer in self.root.buffers.iter_mut() {
            buffer.uri = Some(bin_buffer_name.clone());
        }
        self.bin_buffer_name = Some(bin_buffer_name);
    }

    #[track_caller]
    /// Push a gltf element to the builder
    pub fn push<T>(&mut self, value: T) -> Index<T>
    where
        Self: AsMut<Vec<T>>,
    {
        Index::push(self.as_mut(), value)
    }

    pub fn push_buffer<T>(&mut self, buffer: Vec<T>) -> Index<Buffer> {
        let byte_buffer = vec_to_u8_vec(buffer);
        if self.root.buffers.is_empty() {
            self.root.buffers.push(Buffer {
                byte_length: USize64::from(byte_buffer.len()),
                name: None,
                uri: if self.text_format {
                    self.bin_buffer_name.clone()
                } else {
                    None
                },
                extensions: None,
                extras: Extras::default(),
            })
        } else {
            self.root.buffers[0].byte_length.0 += byte_buffer.len() as u64;
        }
        let index = self.blob.len();
        self.blob.push(byte_buffer);
        Index::new(index as u32)
    }

    pub fn push_buffer_with_view<T>(
        &mut self,
        name: Option<String>,
        buffer: Vec<T>,
        stride: Option<usize>,
    ) -> Index<View> {
        let t_size = core::mem::size_of::<T>();
        let buffer_length = buffer.len() * t_size;
        let buffer = self.push_buffer(buffer);
        self.push_view(View {
            buffer,
            byte_length: USize64::from(buffer_length),
            byte_offset: None,
            byte_stride: Some(Stride(stride.unwrap_or(1) * t_size)),
            extensions: Default::default(),
            extras: Default::default(),
            name,
            target: Some(Checked::Valid(Target::ArrayBuffer)),
        })
    }

    fn get_buffer_offset(&self, buffer: Index<Buffer>) -> u64 {
        self.blob[..buffer.value()]
            .iter()
            .map(|it| it.len())
            .sum::<usize>() as u64
    }

    pub fn push_view(&mut self, view: View) -> Index<View> {
        let mut view = view;
        let buffer = view.buffer;
        let original_offset = view.byte_offset.unwrap_or_default().0;
        view.buffer = Index::new(0);
        view.byte_offset = Some((self.get_buffer_offset(buffer) + original_offset).into());
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

    fn concat_buffers(&self) -> Vec<u8> {
        self.blob
            .iter()
            .flat_map(|it| it.as_slice())
            .cloned()
            .collect()
    }

    #[allow(dead_code)]
    pub fn to_json(&self) -> String {
        json::serialize::to_string(&self.root).expect("Serialization error")
    }

    /// @param out_dir: only for text format. The file in which to write the binary data
    #[allow(dead_code)]
    pub fn write_to_glb<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        assert!(!self.text_format, "Builder is not in binary mode");
        let buffers = self.concat_buffers();
        let buffer_length = buffers.len();

        let json_string = json::serialize::to_string(&self.root).expect("Serialization error");
        let mut json_length = json_string.len();
        align_to_multiple_of_four(&mut json_length);

        let header = Header {
            magic: *b"glTF",
            version: 2,
            length: (json_length + buffer_length)
                .try_into()
                .expect("file size exceeds binary glTF limit"),
        };
        let out = Glb {
            header,
            json: Cow::Owned(json_string.into_bytes()),
            bin: Some(Cow::Owned(to_padded_byte_vector(buffers))),
        };

        out.to_writer(writer)
            .unwrap_or_else(|_| panic!("Unable to write to output"));

        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_to_gltf<W, B>(&self, writer: W, mut bin_writer: B) -> Result<(), String>
    where
        W: std::io::Write,
        B: std::io::Write,
    {
        assert!(self.text_format, "Builder is not in text mode");
        let buffers = self.concat_buffers();
        json::serialize::to_writer(writer, &self.root).expect("Serialization error");
        bin_writer
            .write_all(buffers.as_slice())
            .expect("Error writing binary data");
        Ok(())
    }
}

fn align_to_multiple_of_four(n: &mut usize) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * std::mem::size_of::<T>();
    let byte_capacity = vec.capacity() * std::mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }
    new_vec
}

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

impl_get!(Accessor, accessors);
impl_get!(Animation, animations);
// impl_get!(Buffer, buffers);
// impl_get!(buffer::View, buffer_views);
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

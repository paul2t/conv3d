mod gltf_builder;

use clap::{clap_derive::ValueEnum, Parser};
use glob::glob;
use gltf::json;
use gltf_builder::GltfBuilder;
use json::validation::Checked::Valid;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    fs::{File, OpenOptions},
    io::BufWriter,
    path::Path,
};
use stl_io::IndexedMesh;

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, PartialOrd, Ord)]
enum FileFormat {
    Stl,
    Gltf,
    Glb,
}

fn get_extension(format: FileFormat) -> &'static str {
    match format {
        FileFormat::Stl => "stl",
        FileFormat::Gltf => "gltf",
        FileFormat::Glb => "glb",
    }
}

#[derive(Parser)]
struct App {
    input_files: Vec<String>,

    #[arg(short, long)]
    output_format: FileFormat,
}

fn main() {
    let app = App::parse();

    println!("{:?} {:?}", app.output_format, app.input_files);

    let mut input_files = Vec::new();
    for pattern in app.input_files {
        for entry in glob(&pattern)
            .unwrap_or_else(|_| panic!("Unable to read pattern: {}", &pattern))
            .flatten()
        {
            input_files.push(entry);
        }
    }

    input_files.par_iter().for_each(|path| {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)
            .unwrap_or_else(|_| panic!("Unable to open {}", path.display()));
        let stl = stl_io::read_stl(&mut file)
            .unwrap_or_else(|_| panic!("Unable to parse {}", path.display()));
        println!("Parsed {}", path.display());

        let mut outpath = path.clone();
        outpath.set_extension(get_extension(app.output_format.to_owned()));
        if outpath != *path {
            let gltf =
                if app.output_format == FileFormat::Glb || app.output_format == FileFormat::Gltf {
                    convert_stl_to_gltf(stl, path).unwrap()
                } else {
                    unimplemented!()
                };
            let file = File::create(outpath.clone()).unwrap();
            let writer = BufWriter::new(file);
            if app.output_format == FileFormat::Glb {
                let glb = gltf.to_glb().unwrap();
                glb.to_writer(writer).unwrap();
            } else if app.output_format == FileFormat::Gltf {
                let file = File::create(outpath.clone()).unwrap();
                let writer = BufWriter::new(file);
                let mut gltf = gltf.merge_gltf_buffers().unwrap();
                gltf.set_buffer_uri(
                    0,
                    Some(format!(
                        "{}.bin",
                        outpath
                            .file_stem()
                            .unwrap_or_default()
                            .to_str()
                            .unwrap_or_default()
                    )),
                )
                .unwrap();
                gltf.write_to_gltf(writer).unwrap();
                gltf.write_all_buffers(outpath.parent().unwrap_or(Path::new(".")))
                    .unwrap();
            }

            println!("Output: {}", outpath.display());
        }
    });
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct V3 {
    v: [f32; 3],
}

/// Calculate bounding coordinates of a list of vertices, used for the clipping distance of the model
fn bounding_coords(points: &[V3]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    for point in points {
        let p = point.v;
        for i in 0..3 {
            min[i] = f32::min(min[i], p[i]);
            max[i] = f32::max(max[i], p[i]);
        }
    }
    (min, max)
}

fn convert_stl_to_gltf(
    stl: IndexedMesh,
    input_filename: impl AsRef<Path>,
) -> Result<GltfBuilder, String> {
    let mesh_name = input_filename
        .as_ref()
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();

    let mut gltf = GltfBuilder::new();
    let with_indices = true;

    let (positions, mut normals) = stl
        .vertices
        .iter()
        .map(|it| {
            (
                V3 {
                    v: [it[0], it[1], it[2]],
                },
                V3 { v: [0.0, 0.0, 0.0] },
            )
        })
        .collect::<(Vec<V3>, Vec<V3>)>();

    let mut normals_count = vec![0; normals.len()];
    for face in &stl.faces {
        for vi in face.vertices {
            normals[vi].v[0] += face.normal[0];
            normals[vi].v[1] += face.normal[1];
            normals[vi].v[2] += face.normal[2];
            normals_count[vi] += 1;
        }
    }

    // normalize
    for i in 0..normals.len() {
        let n = normals[i].v;
        let count = normals_count[i] as f32;
        normals[i].v = [n[0] / count, n[1] / count, n[2] / count];
    }

    let (min, max) = bounding_coords(&positions);
    println!("min: {min:?} max: {max:?}");
    let vcount = positions.len();

    let positions_view =
        gltf.push_buffer_with_view(Some("positions".to_string()), positions, None, None);

    let normals_view = gltf.push_buffer_with_view(Some("normals".to_string()), normals, None, None);

    let positions = gltf.push_accessor_vec3(
        Some("positions".to_string()),
        positions_view,
        0,
        vcount,
        Some(min),
        Some(max),
    );
    let normals = gltf.push_accessor_vec3(
        Some("normals".to_string()),
        normals_view,
        3,
        vcount,
        None,
        None,
    );

    let indices = stl
        .faces
        .iter()
        .flat_map(|it| {
            [
                it.vertices[0] as u32,
                it.vertices[1] as u32,
                it.vertices[2] as u32,
            ]
        })
        .collect::<Vec<_>>();
    let nb_indices = indices.len();
    let indices_view =
        gltf.push_buffer_with_view(Some("indices".to_string()), indices, Some(1), None);
    let indices = if with_indices {
        Some(gltf.push_accessor_u32(Some("indices".to_string()), indices_view, 0, nb_indices))
    } else {
        None
    };

    let primitive = json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(Valid(json::mesh::Semantic::Positions), positions);
            map.insert(Valid(json::mesh::Semantic::Normals), normals);
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices,
        material: None,
        mode: Valid(json::mesh::Mode::Triangles),
        targets: None,
    };

    let mesh = gltf.push_mesh(Some(mesh_name), vec![primitive], None);
    let node = gltf.push_node(mesh);
    let scene = gltf.push_scene(vec![node]);
    gltf.set_default_scene(Some(scene));

    Ok(gltf)
}

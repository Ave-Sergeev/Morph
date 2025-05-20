fn main() {
    let proto_gen_dir = "./src/pb";
    let proto_file = "./proto/voice_embeddings.proto";
    let arg = "--experimental_allow_proto3_optional";

    tonic_build::configure()
        .protoc_arg(arg)
        .out_dir(proto_gen_dir)
        .build_client(false)
        .build_server(true)
        .compile_protos(&[proto_file], &["proto"])
        .unwrap_or_else(|err| panic!("Failed to compile protos {err}"));
}

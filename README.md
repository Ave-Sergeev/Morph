## Morph

---

### Description

...

### Local startup

1) To install `Rust` on Unix-like systems (MacOS, Linux, ...) - run the command in the terminal.
   After the download is complete, you will get the latest stable version of Rust for your platform, as well as the latest version of Cargo.

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2) Run the following command in the terminal to verify.

```shell
cargo --version
```

3) Open the project and run the commands.

Check the code to see if it can be compiled (without running it).
```shell
cargo check
```

Build + run the project (in release mode with optimizations).
```shell
cargo run --release
```

UDP: If you have Windows, see [Instructions here](https://forge.rust-lang.org/infra/other-installation-methods.html).

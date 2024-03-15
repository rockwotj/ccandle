extern crate cbindgen;

use std::env;

use cbindgen::{Config, Language};

const HEADER: &'static str = "/*
 * Copyright 2024 Redpanda Data, Inc.
 *
 * Use of this software is governed by the Business Source License
 * included in the file licenses/BSL.md
 *
 * As of the Change Date specified in that file, in accordance with
 * the Business Source License, use of this software will be governed
 * by the Apache License, Version 2.0
 */
 
#pragma once

#ifdef __clang__
#define NONNULL _Nonnull
#else
#define NONNULL
#endif
";

const TRAILER: &'static str = "
#undef NONNULL
";

fn main() {
    eprintln!("=== START CCANDLE ENV VARS ===");
    for (k, v) in env::vars() {
        eprintln!("{k}={v}")
    }
    eprintln!("=== END CCANDLE ENV VARS ===");
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut config = Config::default();
    config.header = Some(HEADER.to_owned());
    config.trailer = Some(TRAILER.to_owned());
    config.namespace = Some("ccandle".to_owned());
    config.language = Language::C;
    config.cpp_compat = true;
    config.usize_is_size_t = true;
    config.style = cbindgen::Style::Tag;
    config.pointer.non_null_attribute = Some("NONNULL".to_owned());
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/ccandle/ccandle.h");
}

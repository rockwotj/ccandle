/*
 * Copyright 2024 Redpanda Data, Inc.
 *
 * Use of this software is governed by the Business Source License
 * included in the file licenses/BSL.md
 *
 * As of the Change Date specified in that file, in accordance with
 * the Business Source License, use of this software will be governed
 * by the Apache License, Version 2.0
 */

use std::{mem, ptr, slice};

mod mistral;

/// A representation of a loaded machine learning model.
pub enum Model {
    Mistral(mistral::TextGeneration),
}

/// Load a model from hugging face by canonical name.
///
/// Supported models:
/// - Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.1
#[no_mangle]
pub unsafe extern "C" fn ccandle_load_model(model_name: UnownedString) -> Option<Box<Model>> {
    let model_name: &str = match model_name.as_str() {
        Ok(s) => s,
        Err(_) => return None,
    };
    if model_name == "mistral" {
        let model = match mistral::TextGeneration::load_from_hugging_face() {
            Ok(m) => m,
            Err(_) => return None,
        };
        return Some(Box::new(Model::Mistral(model)));
    }
    return None;
}

/// Delete a model that has been loaded from `ccandle_load_model`.
#[no_mangle]
pub unsafe extern "C" fn ccandle_delete_model(_: Box<Model>) {}

/// An unowned string in Rust. Rust does not try to delete this data.
#[repr(C)]
pub struct UnownedString {
    pub data: *const u8,
    pub length: usize,
}

impl UnownedString {
    fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        if self.data.is_null() {
            assert!(self.length == 0);
            return Ok("");
        }
        std::str::from_utf8(unsafe { slice::from_raw_parts(self.data, self.length) })
    }
}

/// An owned string in Rust. This object should be deleted using `ccandle_delete_owned_string`.
#[repr(C)]
pub struct OwnedString {
    pub data: *mut u8,
    pub length: usize,
    pub capacity: usize,
}

impl OwnedString {
    pub fn into_string(mut self) -> String {
        let s = unsafe { String::from_raw_parts(self.data, self.length, self.capacity) };
        self.data = ptr::null_mut();
        self.length = 0;
        self.capacity = 0;
        s
    }
}

/// Delete an OwnedString using the Rust allocator.
#[no_mangle]
pub unsafe extern "C" fn ccandle_delete_owned_string(_: Box<OwnedString>) {}

impl From<String> for OwnedString {
    fn from(value: String) -> Self {
        let mut value = value.into_bytes();
        let result = Self {
            data: value.as_mut_ptr(),
            length: value.len(),
            capacity: value.capacity(),
        };
        mem::forget(value);
        result
    }
}

impl Drop for OwnedString {
    fn drop(&mut self) {
        if self.data.is_null() {
            return;
        }
        unsafe {
            Vec::from_raw_parts(self.data, self.length, self.capacity);
        }
        self.data = ptr::null_mut();
        self.length = 0;
        self.capacity = 0;
    }
}

/// Run model
#[no_mangle]
pub unsafe extern "C" fn ccandle_run_model(
    model: &mut Model,
    prompt: UnownedString,
    max_tokens: usize,
) -> Option<Box<OwnedString>> {
    let prompt = match prompt.as_str() {
        Ok(s) => s,
        Err(_) => return None,
    };
    eprintln!("prompt: {}", prompt.escape_default());
    let response = match model {
        Model::Mistral(m) => m.run(prompt, max_tokens),
    };
    eprintln!("response: {}", response.as_ref().unwrap().escape_default());
    match response {
        Ok(s) => Some(Box::new(OwnedString::from(s))),
        Err(_) => None,
    }
}

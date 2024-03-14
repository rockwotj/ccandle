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
 
#pragma once

#ifdef __clang__
#define NONNULL _Nonnull
#else
#define NONNULL
#endif


#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
namespace ccandle {
#endif // __cplusplus

/**
 * A representation of a loaded machine learning model.
 */
struct Model;

/**
 * An unowned string in Rust. Rust does not try to delete this data.
 */
struct UnownedString {
  const uint8_t *data;
  size_t length;
};

/**
 * An owned string in Rust. This object should be deleted using `ccandle_delete_owned_string`.
 */
struct OwnedString {
  uint8_t *data;
  size_t length;
  size_t capacity;
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Load a model from hugging face by canonical name.
 *
 * Supported models:
 * - Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.1
 */
struct Model *ccandle_load_model(struct UnownedString model_name);

/**
 * Delete a model that has been loaded from `ccandle_load_model`.
 */
void ccandle_delete_model(struct Model*NONNULL );

/**
 * Delete an OwnedString using the Rust allocator.
 */
void ccandle_delete_owned_string(struct OwnedString*NONNULL );

/**
 * Run model
 */
struct OwnedString *ccandle_run_model(struct Model *NONNULL model,
                                      struct UnownedString prompt,
                                      size_t max_tokens);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#ifdef __cplusplus
} // namespace ccandle
#endif // __cplusplus


#undef NONNULL

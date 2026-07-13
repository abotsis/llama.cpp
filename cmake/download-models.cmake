get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

# REPO defaults to ggml-org/models; callers may point NAME at a file inside a
# different Hugging Face repo (e.g. a small hybrid test model in its own repo).
if(NOT DEFINED REPO OR REPO STREQUAL "")
    set(REPO "ggml-org/models")
endif()

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ${REPO}...")
endif()

file(DOWNLOAD
    "https://huggingface.co/${REPO}/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()

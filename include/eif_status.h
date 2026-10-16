#ifndef EIF_STATUS_H
#define EIF_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EIF_STATUS_OK = 0,
    EIF_STATUS_ERROR = -1,
    EIF_STATUS_INVALID_ARGUMENT = -2,
    EIF_STATUS_OUT_OF_MEMORY = -3,
    EIF_STATUS_NOT_IMPLEMENTED = -4,
    EIF_STATUS_NOT_SUPPORTED = -5,
    EIF_STATUS_TIMEOUT = -6,
    EIF_STATUS_INTERNAL_ERROR = -7,
    EIF_STATUS_INVALID_LAYER = -8,
    EIF_STATUS_SHAPE_MISMATCH = -9
} eif_status_t;

/**
 * @brief Convert status code to human-readable string.
 *
 * @param status Status code to convert
 * @return Constant string describing the status (never NULL)
 */
static inline const char *eif_status_to_string(eif_status_t status)
{
    switch (status) {
    case EIF_STATUS_OK:
        return "OK";
    case EIF_STATUS_ERROR:
        return "Generic error";
    case EIF_STATUS_INVALID_ARGUMENT:
        return "Invalid argument (NULL pointer or out-of-range)";
    case EIF_STATUS_OUT_OF_MEMORY:
        return "Out of memory (pool allocation failed)";
    case EIF_STATUS_NOT_IMPLEMENTED:
        return "Not implemented";
    case EIF_STATUS_NOT_SUPPORTED:
        return "Not supported in this configuration";
    case EIF_STATUS_TIMEOUT:
        return "Operation timed out";
    case EIF_STATUS_INTERNAL_ERROR:
        return "Internal error (should not happen)";
    case EIF_STATUS_INVALID_LAYER:
        return "Invalid or unsupported layer type";
    case EIF_STATUS_SHAPE_MISMATCH:
        return "Tensor shape mismatch";
    default:
        return "Unknown status code";
    }
}

#ifdef __cplusplus
}
#endif
#endif /* EIF_STATUS_H */

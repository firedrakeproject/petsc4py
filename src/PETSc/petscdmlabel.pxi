# --------------------------------------------------------------------

cdef extern from * nogil:

    int DMLabelGetStratumIS(PetscDMLabel,PetscInt,PetscIS*)
    int DMLabelSetValue(DMLabel,PetscInt,PetscInt)

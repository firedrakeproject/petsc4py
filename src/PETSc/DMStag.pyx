# --------------------------------------------------------------------

class DMStagStencilType(object):
    STAR = DMSTAG_STENCIL_STAR
    BOX  = DMSTAG_STENCIL_BOX
    NONE = DMSTAG_STENCIL_NONE

class DMStagStencilLocation(object):
    NULLLOC          = DMSTAG_NULL_LOCATION
    BACK_DOWN_LEFT   = DMSTAG_BACK_DOWN_LEFT
    BACK_DOWN        = DMSTAG_BACK_DOWN
    BACK_DOWN_RIGHT  = DMSTAG_BACK_DOWN_RIGHT
    BACK_LEFT        = DMSTAG_BACK_LEFT
    BACK             = DMSTAG_BACK
    BACK_RIGHT       = DMSTAG_BACK_RIGHT
    BACK_UP_LEFT     = DMSTAG_BACK_UP_LEFT
    BACK_UP          = DMSTAG_BACK_UP
    BACK_UP_RIGHT    = DMSTAG_BACK_UP_RIGHT
    DOWN_LEFT        = DMSTAG_DOWN_LEFT
    DOWN             = DMSTAG_DOWN
    DOWN_RIGHT       = DMSTAG_DOWN_RIGHT
    LEFT             = DMSTAG_LEFT
    ELEMENT          = DMSTAG_ELEMENT
    RIGHT            = DMSTAG_RIGHT
    UP_LEFT          = DMSTAG_UP_LEFT
    UP               = DMSTAG_UP
    UP_RIGHT         = DMSTAG_UP_RIGHT
    FRONT_DOWN_LEFT  = DMSTAG_FRONT_DOWN_LEFT
    FRONT_DOWN       = DMSTAG_FRONT_DOWN
    FRONT_DOWN_RIGHT = DMSTAG_FRONT_DOWN_RIGHT
    FRONT_LEFT       = DMSTAG_FRONT_LEFT
    FRONT            = DMSTAG_FRONT
    FRONT_RIGHT      = DMSTAG_FRONT_RIGHT
    FRONT_UP_LEFT    = DMSTAG_FRONT_UP_LEFT
    FRONT_UP         = DMSTAG_FRONT_UP
    FRONT_UP_RIGHT   = DMSTAG_FRONT_UP_RIGHT
    
# ADD DMStagStencil


# --------------------------------------------------------------------

cdef class DMStag(DM):

    StencilType       = DMStagStencilType
    StencilLocation   = DMStagStencilLocation
# ADD DMStagStencil

    def create(self,dim,dofs,sizes,boundary_types,stencil_type,stencil_width,proc_sizes=None,ownership_ranges=None,comm=None):
        
        # ndim
        cdef PetscInt ndim = asInt(dim)
        
        # sizes
        cdef tuple gsizes = tuple(sizes)
        cdef PetscInt nsizes=PETSC_DECIDE, M=1, N=1, P=1
        nsizes = asStagDims(gsizes, &M, &N, &P)
        assert(nsizes==ndim)
           
        # dofs
        cdef tuple cdofs = tuple(dofs)
        cdef PetscInt ndofs=PETSC_DECIDE, dof0=1, dof1=0, dof2=0, dof3=0
        ndofs = asDofs(cdofs, &dof0, &dof1, &dof2, &dof3)
        assert(ndofs==ndim+1)

        # boundary types
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        asBoundary(boundary_types, &btx, &bty, &btz)
        
        # stencil
        cdef PetscInt swidth = asInt(stencil_width)
        cdef PetscDMStagStencilType stype = asStagStencil(stencil_type)

        # comm
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)

        # proc sizes
        cdef object psizes = proc_sizes
        cdef PetscInt nprocs=PETSC_DECIDE, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        if proc_sizes is not None:
            nprocs = asStagDims(psizes, &m, &n, &p)
            assert(nprocs==ndim)

        # ownership ranges
        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        if ownership_ranges is not None:
            nranges = asStagOwnershipRanges(ownership_ranges, ndim, &m, &n, &p, &lx, &ly, &lz)       
            
        # create
        cdef PetscDM newda = NULL
        if dim == 1:
            CHKERR( DMStagCreate1d(ccomm, btx, M, dof0, dof1, stype, swidth, lx, &newda) )
        if dim == 2:
            CHKERR( DMStagCreate2d(ccomm, btx, bty, M, N, m, n, dof0, dof1, dof2, stype, swidth, lx, ly, &newda) )
        if dim == 3:
            CHKERR( DMStagCreate3d(ccomm, btx, bty, btz, M, N, P, m, n, p, dof0, dof1, dof2, dof3, stype, swidth, lx, ly, lz, &newda) )
        return self
            
    # Setters


    def setStencilWidth(self,swidth):
        raise NotImplementedError('DMStagSetStencilWidth not yet implemented in petsc')
#        cdef PetscInt sw = asInt(swidth)
#        CHKERR( DMStagSetStencilWidth(self.dm, sw) )

    def setGhostType(self, ghosttype):
        raise NotImplementedError('DMStagSetGhostType not yet implemented in petsc4py')
#        cdef PetscDMStagStencilType stype = asStagStencil(ghosttype)
#        CHKERR( DMStagSetGhostType(self.dm, stype) )

    def setBoundaryTypes(self, boundary_types):
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        asBoundary(boundary_types, &btx, &bty, &btz)
        CHKERR( DMStagSetBoundaryTypes(self.dm, btx, bty, btz) )    
        
    def setDof(self, dofs):
        cdef tuple gdofs = tuple(dofs)
        cdef PetscInt gdim=PETSC_DECIDE, dof0=1, dof1=0, dof2=0, dof3=0
        gdim = asDofs(gdofs, &dof0, &dof1, &dof2, &dof3)
        CHKERR( DMStagSetDOF(self.dm, dof0, dof1, dof2, dof3) )
        
    def setGlobalSizes(self, sizes):
        cdef tuple gsizes = tuple(sizes)
        cdef PetscInt gdim=PETSC_DECIDE, M=1, N=1, P=1
        gdim = asStagDims(gsizes, &M, &N, &P)
        CHKERR( DMStagSetGlobalSizes(self.dm, M, N, P) )
        
    def setProcSizes(self, sizes):
        cdef tuple psizes = tuple(sizes)
        cdef PetscInt pdim=PETSC_DECIDE, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        pdim = asStagDims(psizes, &m, &n, &p)
        CHKERR( DMStagSetNumRanks(self.dm, m, n, p) )

    def setOwnershipRanges(self, ranges):
        raise NotImplementedError('DMStagSetOwnershipRanges not yet implemented in petsc4py')
#        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
#        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
#        CHKERR( DMGetDimension(self.dm, &dim) )
#        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
#        ownership_ranges = asStagOwnershipRanges(ranges, dim, &m, &n, &p, &lx, &ly, &lz)
#        CHKERR( DMStagSetOwnershipRanges(self.dm, lx, ly, lz) )





    
    
      
    
    
    # Getters
    
    def getEntriesPerElement(self):
        cdef PetscInt epe=0
        CHKERR( DMStagGetEntriesPerElement(self.dm, &epe) )
        return toInt(epe)
    
    def getStencilWidth(self):
        cdef PetscInt swidth=0
        CHKERR( DMStagGetStencilWidth(self.dm, &swidth) )
        return toInt(swidth)

    def getDof(self):
        cdef PetscInt dim=0, dof0=0, dof1=0, dof2=0, dof3=0
        CHKERR( DMStagGetDOF(self.dm, &dof0, &dof1, &dof2, &dof3) )
        CHKERR( DMGetDimension(self.dm, &dim) )
        return toDofs(dim+1,dof0,dof1,dof2,dof3)

    def getCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0, nExtrax=0, nExtray=0, nExtraz=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetCorners(self.dm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz) )
        return (asInt(x), asInt(y), asInt(z))[:<Py_ssize_t>dim], (asInt(m), asInt(n), asInt(p))[:<Py_ssize_t>dim], (asInt(nExtrax), asInt(nExtray), asInt(nExtraz))[:<Py_ssize_t>dim]

    def getGhostCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetGhostCorners(self.dm, &x, &y, &z, &m, &n, &p) )
        return (asInt(x), asInt(y), asInt(z))[:<Py_ssize_t>dim], (asInt(m), asInt(n), asInt(p))[:<Py_ssize_t>dim]

    def getLocalSizes(self):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetLocalSizes(self.dm, &m, &n, &p) )
        return toStagDims(dim, m, n, p)

    def getGlobalSizes(self):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetGlobalSizes(self.dm, &m, &n, &p) )
        return toStagDims(dim, m, n, p)
        
    def getProcSizes(self):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
        return toStagDims(dim, m, n, p)
        
    def getGhostType(self):
        raise NotImplementedError('DMStagGetGhostType not yet implemented in petsc')
#        cdef PetscDMStagStencilType stype = DMSTAG_STENCIL_BOX
#        CHKERR( DMStagGetGhostType(self.dm, &stype) )
#        return toStagStencil(stype)

    def getOwnershipRanges(self):
        raise NotImplementedError('DMStagGetOwnershipRanges not yet implemented in petsc')
#        cdef PetscInt dim=0, m=0, n=0, p=0
#        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
#        CHKERR( DMGetDimension(self.dm, &dim) )
#        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
#        CHKERR( DMStagGetOwnershipRanges(self.dm, &lx, &ly, &lz) )
#        return toStagOwnershipRanges(dim, m, n, p, lx, ly, lz)

    def getBoundaryTypes(self):
        cdef PetscInt dim=0
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetBoundaryTypes(self.dm, &btx, &bty, &btz) )
        return toStagDims(dim, btx, bty, btz)

    def getIsFirstRank(self):
        cdef PetscBool rank0=PETSC_FALSE, rank1=PETSC_FALSE, rank2=PETSC_FALSE
        cdef PetscInt dim=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetIsFirstRank(self.dm, &rank0, &rank1, &rank2) )
        return toStagDims(dim, rank0, rank1, rank2)
        
    def getIsLaskRank(self):
        cdef PetscBool rank0=PETSC_FALSE, rank1=PETSC_FALSE, rank2=PETSC_FALSE
        cdef PetscInt dim=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetIsLastRank(self.dm, &rank0, &rank1, &rank2) )
        return toStagDims(dim, rank0, rank1, rank2)
        
        
    
    
    
    
    
    # Coordinate-related functions

    def setUniformCoordinatesExplicit(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMStagSetUniformCoordinatesExplicit(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax) )        
        
    def setUniformCoordinatesProduct(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMStagSetUniformCoordinatesProduct(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax) )        

        
    def setUniformCoordinates(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMStagSetUniformCoordinates(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax) )        

    def setCoordinateDMType(self, dmtype):
        cdef const_char *cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        CHKERR( DMStagSetCoordinateDMType(self.dm, cval) )

    
    
    # Location slot related functions

    def getLocationSlot(self, loc, c):
        cdef PetscInt slot=0
        cdef PetscInt comp=asInt(c)
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR( DMStagGetLocationSlot(self.dm, sloc, comp, &slot) ) 
        return toInt(slot)

    def get1DCoordinateLocationSlot(self, loc):
        cdef PetscInt slot=0
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR( DMStagGet1dCoordinateLocationSlot(self.dm, sloc, &slot) ) 
        return toInt(slot)
        
    def getLocationDof(self, loc):
        cdef PetscInt dof=0
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR( DMStagGetLocationDOF(self.dm, sloc, &dof) ) 
        return toInt(slot)
        

        


        
        
        

    



    
 
    
    
    
    # Random other functions
    
    def migrateVec(self, Vec vec, DM dmTo, Vec vecTo):
        CHKERR( DMStagMigrateVec(self.dm, vec.vec, dmTo.dm, vecTo.vec ) )
        
    def createCompatibleDMStag(self, dofs, newdm=None):
        raise NotImplementedError('DMStagCreateCompatibleDMStag not yet implemented in petsc4py')
        # int DMStagCreateCompatibleDMStag(PetscDM dm,PetscInt dof0,PetscInt dof1,PetscInt dof2,PetscInt dof3,PetscDM *newdm)
        
    def VecSplitToDMDA(self, vec, loc, c, pda=None, pdavec=None):
        raise NotImplementedError('DMStagVecSplitToDMDA not yet implemented in petsc4py')
        # int DMStagVecSplitToDMDA(PetscDM dm,PetscVec vec,PetscDMStagStencilLocation loc,PetscInt c,PetscDM *pda,PetscVec *pdavec)'

    def getVecArray(self, Vec vec):
        raise NotImplementedError('getVecArray for DMSStag not yet implemented in petsc4py')
        # Should return an error if Vec is not a local DMStag vector

# THESE ARE NOT ACTUALLY WRAPPED
# INSTEAD WE USE A _Vec_Buffer and _DMStag_Vec_Buffer...
#PetscErrorCode DMStagVecGetArrayDOF(DM dm,Vec vec,void *array)
#PetscErrorCode DMStagVecGetArrayDOFRead(DM dm,Vec vec,void *array)
#PetscErrorCode DMStagVecRestoreArrayDOF(DM dm,Vec vec,void *array)
#PetscErrorCode DMStagVecRestoreArrayDOFRead(DM dm,Vec vec,void *array)
#PetscErrorCode DMStagGet1dCoordinateArraysDOFRead(DM dm,void* arrX,void* arrY,void* arrZ)
#PetscErrorCode DMStagRestore1dCoordinateArraysDOFRead(DM dm,void *arrX,void *arrY,void *arrZ)




# This needs to go into Mat/Vec actually
#PetscErrorCode DMStagMatSetValuesStencil(DM dm,Mat mat,PetscInt nRow,const DMStagStencil *posRow,PetscInt nCol,const DMStagStencil *posCol,const PetscScalar *val,InsertMode insertMode)
#PetscErrorCode DMStagVecGetValuesStencil(DM dm,Vec vec,PetscInt n,const DMStagStencil *pos,PetscScalar *val)
#PetscErrorCode DMStagVecSetValuesStencil(DM dm,Vec vec,PetscInt n,const DMStagStencil *pos,const PetscScalar *val,InsertMode insertMode)
# NEED TO ADD DMStagStencil



    property dim:
        def __get__(self):
            return self.getDim()

    property dofs:
        def __get__(self):
            return self.getDof()
    
    property entries_per_element:
        def __get__(self):
            return self.getEntriesPerElement()
            
    property global_sizes:
        def __get__(self):
            return self.getGlobalSizes()
            
    property local_sizes:
        def __get__(self):
            return self.getLocalSizes()

    property proc_sizes:
        def __get__(self):
            return self.getProcSizes()

    property boundary_types:
        def __get__(self):
            return self.getBoundaryTypes()

    property stencil_type:
        def __get__(self):
            return self.getGhostType()

    property stencil_width:
        def __get__(self):
            return self.getStencilWidth()

    property corners:
        def __get__(self):
            return self.getCorners()

    property ghost_corners:
        def __get__(self):
            return self.getGhostCorners()


# --------------------------------------------------------------------

del DMStagStencilType
del DMStagStencilLocation

# --------------------------------------------------------------------

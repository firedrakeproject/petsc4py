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


    def create(self, dim=None, dofs=None,
               sizes=None, proc_sizes=None, boundary_type=None,
               stencil_type=None, stencil_width=None,
               bint setup=True, ownership_ranges=None, comm=None):
        #
        cdef object arg = None
        try: arg = tuple(dim)
        except TypeError: pass
        else: dim, sizes = None, arg
        #
        cdef PetscInt ndim = PETSC_DECIDE
        cdef PetscInt M = 1, m = PETSC_DECIDE, *lx = NULL
        cdef PetscInt N = 1, n = PETSC_DECIDE, *ly = NULL
        cdef PetscInt P = 1, p = PETSC_DECIDE, *lz = NULL
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        cdef PetscDMStagStencilType stype  = DMSTAG_STENCIL_BOX
        cdef PetscInt             swidth = PETSC_DECIDE
        
        # grid and proc sizes
        cdef object gsizes = sizes
        cdef object psizes = proc_sizes
        cdef PetscInt gdim = PETSC_DECIDE
        cdef PetscInt pdim = PETSC_DECIDE
        if sizes is not None:
            gdim = asStagDims(gsizes, &M, &N, &P)
        if psizes is not None:
            pdim = asStagDims(psizes, &m, &n, &p)
        if gdim>=0 and pdim>=0:
            assert gdim == pdim
            
        # dofs
        cdef PetscInt dof0=1, dof1=0, dof2=0, dof3=0
        cdef object pdofs = dofs
        if dofs is not None:
            asDofs(pdofs, &dof0, &dof1, &dof2, &dof3)

        # dim
        if dim is not None: ndim = asInt(dim)
        if ndim==PETSC_DECIDE: ndim = gdim
        
        # vertex distribution
        if ownership_ranges is not None:
            ownership_ranges = asStagOwnershipRanges(ownership_ranges,
                                                 ndim, &m, &n, &p,
                                                 &lx, &ly, &lz)
                                                 
        # periodicity, stencil type & width
        if boundary_type is not None:
            asBoundary(boundary_type, &btx, &bty, &btz)
        if stencil_type is not None:
            stype = asStagStencil(stencil_type)
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        if setup and swidth == PETSC_DECIDE: swidth = 0
        
        # create the DMStag object
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newda = NULL
        CHKERR( DMStagCreateND(ccomm, ndim, dof0, dof1, dof2, dof3,
                             M, N, P, m, n, p, lx, ly, lz,
                             btx, bty, btz, stype, swidth,
                             &newda) )
        if setup and ndim > 0: CHKERR( DMSetUp(newda) )
        PetscCLEAR(self.obj); self.dm = newda
        return self

    
    
    
    
    # Setters


#    def setStencilWidth(self,swidth):
#        sw = asInt(dof0)
#        CHKERR( DMStagSetStencilWidth(self.dm, sw) )
       
#    def setGhostType(self, ghosttype):
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
        cdef PetscInt gdim = PETSC_DECIDE
        cdef PetscInt dof0 = 1
        cdef PetscInt dof1 = 0
        cdef PetscInt dof2 = 0
        cdef PetscInt dof3 = 0
        gdim = asDofs(gdofs, &dof0, &dof1, &dof2, &dof3)
        CHKERR( DMStagSetDOF(self.dm, dof0, dof1, dof2, dof3) )
        
    def setGlobalSizes(self, sizes):
        cdef tuple gsizes = tuple(sizes)
        cdef PetscInt gdim = PETSC_DECIDE
        cdef PetscInt M = 1
        cdef PetscInt N = 1
        cdef PetscInt P = 1
        gdim = asStagDims(gsizes, &M, &N, &P)
        cdef PetscInt dim = PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        if dim == PETSC_DECIDE:
            CHKERR( DMSetDimension(self.dm, gdim) )
        CHKERR( DMStagSetGlobalSizes(self.dm, M, N, P) )
        
    def setProcSizes(self, sizes):
        cdef tuple psizes = tuple(sizes)
        cdef PetscInt pdim = PETSC_DECIDE
        cdef PetscInt m = PETSC_DECIDE
        cdef PetscInt n = PETSC_DECIDE
        cdef PetscInt p = PETSC_DECIDE
        pdim = asStagDims(psizes, &m, &n, &p)
        cdef PetscInt dim = PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        if dim == PETSC_DECIDE:
            CHKERR( DMSetDimension(self.dm, pdim) )
        CHKERR( DMStagSetNumRanks(self.dm, m, n, p) )

#    def setOwnershipRanges(self, ranges):
#        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
#        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
#        CHKERR( DMGetDimension(self.dm, &dim) )
#        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
#        ownership_ranges = asStagOwnershipRanges(ranges, dim, &m, &n, &p, &lx, &ly, &lz)
#        CHKERR( DMStagSetOwnershipRanges(self.dm, &lx, &ly, &lz) )





    
    
      
    
    
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
        
#    def getGhostType(self):
#        cdef PetscDMStagStencilType stype = DMSTAG_STENCIL_BOX
#        CHKERR( DMStagGetGhostType(self.dm, &stype) )

#    def getOwnershipRanges(self):
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
        pass
        
    #int DMStagSetCoordinateDMType(PetscDM dm,PetscDMType dmtype)   


    
    
    
    
    
    # Location slot related functions
    
    def getLocationSlot(self):
        pass
    

    def getLocationDof(self):
        pass
        
    def get1DCoordinateLocationSlot(self):
        pass
        
    #int DMStagGetLocationSlot(PetscDM dm,PetscDMStagStencilLocation loc,PetscInt c,PetscInt *slot)
    #int DMStagGetLocationDOF(PetscDM dm,PetscDMStagStencilLocation loc,PetscInt *dof)
    #int DMStagGet1dCoordinateLocationSlot(PetscDM dm,PetscDMStagStencilLocation loc,PetscInt *slot)
        
        
        

    



    
 
    
    
    
    # Random other functions
    
    def migrateVec(self, Vec vec, DM dmTo, Vec vecTo):
        CHKERR( DMStagMigrateVec(self.dm, vec.vec, dmTo.dm, vecTo.vec ) )
        
    def createCompatibleDMStag(self, dofs, newdm=None):
        pass
    def VecSplitToDMDA(self, vec, loc, c, pda=None, pdavec=None):
        pass

        
    #int DMStagCreateCompatibleDMStag(PetscDM dm,PetscInt dof0,PetscInt dof1,PetscInt dof2,PetscInt dof3,PetscDM *newdm)
    #int DMStagVecSplitToDMDA(PetscDM dm,PetscVec vec,PetscDMStagStencilLocation loc,PetscInt c,PetscDM *pda,PetscVec *pdavec)'










# NEED TO ADD VEC GET ARRAY
# NEED TO ADD VEC/MAT SET VALUES STENCIL
# NEED TO ADD DMStagStencil
# NEED TO ADD SOME COORDINATES STUFF

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

#    property stencil_type:
#        def __get__(self):
#            return self.getGhostType()

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

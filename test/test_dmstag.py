from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestDMStag(object):

    COMM = PETSc.COMM_WORLD
    BOUNDARY = PETSc.DM.BoundaryType.NONE
    DOF = (1,0,0,0)
    STENCIL = PETSc.DMStag.StencilType.BOX
    SWIDTH = 1

# ADD TESTS IN HERE FOR PROCSIZES AND OWNERSHIP RANGES

    def setUp(self):
        self.da = PETSc.DMStag().create(len(self.SIZES),
                                      self.DOF,
                                      self.SIZES,
                                      self.BOUNDARY,
                                      self.STENCIL,
                                      self.SWIDTH,
                                      comm=self.COMM)

    def tearDown(self):
        self.da = None

    def testGetInfo(self):
        dim = self.da.getDim()
        dof = self.da.getDof()
        gsizes = self.da.getGlobalSizes()
        psizes = self.da.getProcSizes()
        boundary = self.da.getBoundaryTypes()
        stencil_type = self.da.getGhostType()
        stencil_width = self.da.getStencilWidth()
        entries_per_element = self.da.getEntriesPerElement()
        self.assertEqual(dim, len(self.SIZES))
        self.assertEqual(dof, self.DOF)
        if dim == 1: self.assertEqual(entries_per_element, dof[0] + dof[1])
        if dim == 2: self.assertEqual(entries_per_element, dof[0] + 2*dof[1] + dof[2])
        if dim == 2: self.assertEqual(entries_per_element, dof[0] + 3*dof[1] + 3*dof[2] + dof[3])
        self.assertEqual(gsizes, tuple(self.SIZES))
        self.assertEqual(boundary, self.BOUNDARY)
        self.assertEqual(stencil_type, self.STENCIL)
        self.assertEqual(stencil_width, self.SWIDTH)
    
    def testCorners(self):
        dim = self.da.getDim()
        lsizes = self.da.getLocalSizes()
        starts, sizes, nextra  = self.da.getCorners()
        isLastRank = self.da.getIsLastRank()
        isFirstRank = self.da.getIsFirstRank()
        self.assertEqual(dim, len(starts))
        self.assertEqual(dim, len(sizes))
        self.assertEqual(dim, len(lsizes))
        self.assertEqual(dim, len(nextra))
        self.assertEqual(dim, len(isLastRank))
        self.assertEqual(dim, len(isFirstRank))
        for i in range(dim):
            self.assertEqual(sizes[i], lsizes[i])
            if isLastRank[i]: self.assertEqual(nextra[i],1)
            if (nextra[i]==1): self.assertEqual(isLastRank[i],True)
            

    def testOwnershipRanges(self):
        dim = self.da.getDim()
        ownership_ranges = self.da.getOwnershipRanges()
        procsizes = self.da.getProcSizes()
        self.assertEqual(len(procsizes), len(ownership_ranges))
        for i,m in enumerate(procsizes):
            self.assertEqual(m, len(ownership_ranges[i]))

    def testCoordinates(self):
        
# ADD A TEST HERE FOR SETTING CoordinateDMType
# AND For setUniformCoordinates actually respecting this!

        self.da.setUniformCoordinatesExplicit(0,1,0,1,0,1)
# ACTUALL CHECK THIS TYPE
        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        c.destroy()
# ACTUALL CHECK THIS TYPE
        cda = self.da.getCoordinateDM()
        cda.destroy()
        gc = self.da.getCoordinatesLocal()
        gc.destroy()

        self.da.setUniformCoordinatesProduct(0,1,0,1,0,1)
# ACTUALL CHECK THIS TYPE
        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        c.destroy()
# ACTUALL CHECK THIS TYPE
        cda = self.da.getCoordinateDM()
        cda.destroy()
        gc = self.da.getCoordinatesLocal()
        gc.destroy()


    def testCreateVecMat(self):
        vg = self.da.createGlobalVec()
        vl = self.da.createLocalVec()
        mat = self.da.createMat()
        self.assertTrue(mat.getType() in ('aij', 'seqaij', 'mpiaij'))
        vg.set(1.0)
        self.da.globalToLocal(vg,vl)
        self.assertEqual(vl.max()[1], 1.0)
        self.assertTrue (vl.min()[1] in (1.0, 0.0))
        vl2 = self.da.createLocalVec()
        self.da.localToLocal(vl,vl2)
        self.assertEqual(vl2.max()[1], 1.0)
        self.assertTrue (vl2.min()[1] in (1.0, 0.0))
        NONE = PETSc.DM.BoundaryType.NONE
        s = self.da.stencil_width
        btype = self.da.boundary_type
        psize = self.da.proc_sizes
        for b, p in zip(btype, psize):
            if b != NONE and p == 1: return
        vg2 = self.da.createGlobalVec()
        self.da.localToGlobal(vl2,vg2)

    def testGetVec(self):
        vg = self.da.getGlobalVec()
        vl = self.da.getLocalVec()
        try:
            vg.set(1.0)
            self.assertEqual(vg.max()[1], 1.0)
            self.assertEqual(vg.min()[1], 1.0)
            self.da.globalToLocal(vg,vl)
            self.assertEqual(vl.max()[1], 1.0)
            self.assertTrue (vl.min()[1] in (1.0, 0.0))
            vl.set(2.0)
            NONE = PETSc.DM.BoundaryType.NONE
            s = self.da.stencil_width
            btype = self.da.boundary_type
            psize = self.da.proc_sizes
            for b, p in zip(btype, psize):
                if b != NONE and p == 1: return
            self.da.localToGlobal(vl,vg)
            self.assertEqual(vg.max()[1], 2.0)
            self.assertTrue (vg.min()[1] in (2.0, 0.0))
        finally:
            self.da.restoreGlobalVec(vg)
            self.da.restoreLocalVec(vl)

    def testGetOther(self):
        ao = self.da.getAO()
        lgmap = self.da.getLGMap()
        l2g, g2l = self.da.getScatter()

# ADD TESTS for all the SETTERS
#setGhostType
#setStencilWidth
#setBoundaryTypes
#setDof
#setGlobalSizes
#setProcSizes
#setOwnershipRanges
# Do this via 2 modes of DMStag creation
# 1) DMStag().create()
# 2) DM().create(), DM().setType(), DM().setDim(), etc.
# Then checking equality!

# ADD TESTS FOR migrateVec, createCompatibleDMStag, VecSplitToDMDA

# ADD TEST FOR vecGetArray

# ADD TESTS FOR LOCATION SLOT AND LOCATION DOF
# ADD TEST FOR MAT and VEC SETVALUESSTENCIL

GHOSTED  = PETSc.DM.BoundaryType.GHOSTED
PERIODIC = PETSc.DM.BoundaryType.PERIODIC
NONE = PETSc.DM.BoundaryType.NONE

SCALE = 4

class BaseTestDMStag_1D(BaseTestDMStag):
    SIZES = [100*SCALE,]
    BOUNDARY = [NONE,]

class BaseTestDMStag_2D(BaseTestDMStag):
    SIZES = [9*SCALE,11*SCALE]
    BOUNDARY = [NONE,NONE]

class BaseTestDMStag_3D(BaseTestDMStag):
    SIZES = [6*SCALE,7*SCALE,8*SCALE]
    BOUNDARY = [NONE,NONE,NONE]

# --------------------------------------------------------------------

# ADD MORE SPOT CHECKS HERE!
class TestDMStag_1D(BaseTestDMStag_1D, unittest.TestCase):
    pass
class TestDMStag_1D_W0(TestDMStag_1D):
    SWIDTH = 0
class TestDMStag_1D_W0_N10(TestDMStag_1D):
    SWIDTH = 0
    DOF = (1,0)
class TestDMStag_1D_W0_N11(TestDMStag_1D):
    SWIDTH = 0
    DOF = (1,1)
class TestDMStag_1D_W0_N12(TestDMStag_1D):
    SWIDTH = 0
    DOF = (1,2)
class TestDMStag_1D_W2(TestDMStag_1D):
    SWIDTH = 2
class TestDMStag_1D_W2_N10(TestDMStag_1D):
    SWIDTH = 2
    DOF = (1,0)
class TestDMStag_1D_W2_N11(TestDMStag_1D):
    SWIDTH = 2
    DOF = (1,1)
class TestDMStag_1D_W2_N12(TestDMStag_1D):
    SWIDTH = 2
    DOF = (1,2)

# ADD MORE SPOT CHECKS HERE!
class TestDMStag_2D(BaseTestDMStag_2D, unittest.TestCase):
    pass
class TestDMStag_2D_W0(TestDMStag_2D):
    SWIDTH = 0
class TestDMStag_2D_W0_N112(TestDMStag_2D):
    DOF = (1,1,2)
    SWIDTH = 0
class TestDMStag_2D_W2(TestDMStag_2D):
    SWIDTH = 2
class TestDMStag_2D_W2_N112(TestDMStag_2D):
    DOF = (1,1,2)
    SWIDTH = 2
class TestDMStag_2D_PXY(TestDMStag_2D):
    SIZES = [13*SCALE,17*SCALE]
    DOF = (1,1,2)
    SWIDTH = 5
    BOUNDARY = (PERIODIC,)*2
class TestDMStag_2D_GXY(TestDMStag_2D):
    SIZES = [13*SCALE,17*SCALE]
    DOF = (1,1,2)
    SWIDTH = 5
    BOUNDARY = (GHOSTED,)*2

# ADD MORE SPOT CHECKS HERE!
class TestDMStag_3D(BaseTestDMStag_3D, unittest.TestCase):
    pass
class TestDMStag_3D_W0(TestDMStag_3D):
    SWIDTH = 0
class TestDMStag_3D_W0_N1123(TestDMStag_3D):
    DOF = (1,1,2,3)
    SWIDTH = 0
class TestDMStag_3D_W2(TestDMStag_3D):
    SWIDTH = 2
class TestDMStag_3D_W2_N1123(TestDMStag_3D):
    DOF = (1,1,2,3)
    SWIDTH = 2
class TestDMStag_3D_PXYZ(TestDMStag_3D):
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOF = (1,1,2,3)
    SWIDTH = 3
    BOUNDARY = (PERIODIC,)*3
class TestDMStag_3D_GXYZ(TestDMStag_3D):
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOF = (1,1,2,3)
    SWIDTH = 3
    BOUNDARY = (GHOSTED,)*3

# --------------------------------------------------------------------

DIM = (1,2,3)
DOF0 = (0,1,2,3)
DOF1 = (0,1,2,3)
DOF2 = (0,1,2,3)
DOF3 = (0,1,2,3)
BOUNDARY_TYPE = (PETSc.DM.BoundaryType.NONE,PETSc.DM.BoundaryType.GHOSTED,PETSc.DM.BoundaryType.PERIODIC)
STENCIL_TYPE  = (PETSc.DMStag.StencilType.NONE,PETSc.DMStag.StencilType.STAR,PETSc.DMStag.StencilType.BOX)
STENCIL_WIDTH = (0,1,2,3)

class TestDMStagCreate(unittest.TestCase):
    pass
counter = 0
for dim in DIM:
    for dof0 in DOF0:
        for dof1 in DOF1:
            for dof2 in DOF2:
                if dim == 1 and dof2 > 0: continue
                for dof3 in DOF3:
                    if dim == 2 and dof3 > 0: continue
                    dof = [dof0,dof1,dof2,dof3][:dim+1]
                    for boundary in BOUNDARY_TYPE:
                        for stencil in STENCIL_TYPE:
                            for width in STENCIL_WIDTH:
                                kargs = dict(dim=dim, dof=dof,boundary_type=boundary,stencil_type=stencil,stencil_width=width)
                                def testCreate(self,kargs=kargs):
                                    kargs = dict(kargs)
                                    da = PETSc.DMStag().create(kargs['dim'],kargs['dof'],[8*SCALE,]*kargs['dim'],[kargs['boundary_type'],]*kargs['dim'],kargs['stencil_type'],kargs['stencil_width'])
                                    da.destroy()
                                setattr(TestDMStagCreate,
                                        "testCreate%04d"%counter,
                                        testCreate)
                                del testCreate
                                counter += 1
del counter, dim, dof, dof0, dof1, dof2, dof3, boundary, stencil, width

# --------------------------------------------------------------------

# if PETSc.COMM_WORLD.getSize() > 1:
    # del TestDMStag_1D_W0
    # del TestDMStag_2D_W0, TestDMStag_2D_W0_N2
    # del TestDMStag_3D_W0, TestDMStag_3D_W0_N2

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------

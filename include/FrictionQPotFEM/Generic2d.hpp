/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_GENERIC2D_HPP
#define FRICTIONQPOTFEM_GENERIC2D_HPP

#include "Generic2d.h"

namespace FrictionQPotFEM {
namespace Generic2d {

inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("frictionqpotfem=" + version());
    ret.push_back("gmatelastoplasticqpot=" + GMatElastoPlasticQPot::version());
    ret.push_back("gmattensor=" + GMatTensor::version());
    ret.push_back("goosefem=" + GooseFEM::version());
    ret.push_back("qpot=" + QPot::version());

    ret.push_back(
        "xtensor=" + detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

#if defined(GOOSEFEM_EIGEN) || defined(EIGEN_WORLD_VERSION)
    ret.push_back(
        "eigen=" + detail::unquote(std::string(QUOTE(EIGEN_WORLD_VERSION))) + "." +
        detail::unquote(std::string(QUOTE(EIGEN_MAJOR_VERSION))) + "." +
        detail::unquote(std::string(QUOTE(EIGEN_MINOR_VERSION))));
#endif

#ifdef XSIMD_VERSION_MAJOR
    ret.push_back(
        "xsimd=" + detail::unquote(std::string(QUOTE(XSIMD_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_PATCH))));
#endif

#ifdef XTL_VERSION_MAJOR
    ret.push_back(
        "xtl=" + detail::unquote(std::string(QUOTE(XTL_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_PATCH))));
#endif

#ifdef XTENSOR_PYTHON_VERSION_MAJOR
     ret.push_back(
         "xtensor-python=" + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MAJOR))) + "." +
         detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MINOR))) + "." +
         detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_PATCH))));
 #endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

template <class C, class E, class L>
inline System::System(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

template <class C, class E, class L>
inline void System::initSystem(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    m_coor = coor;
    m_conn = conn;
    m_dofs = dofs;
    m_iip = iip;
    m_elem_elas = elem_elastic;
    m_elem_plas = elem_plastic;

    m_nnode = m_coor.shape(0);
    m_ndim = m_coor.shape(1);
    m_nelem = m_conn.shape(0);
    m_nne = m_conn.shape(1);

    m_nelem_elas = m_elem_elas.size();
    m_nelem_plas = m_elem_plas.size();
    m_set_elas = !m_nelem_elas;
    m_set_plas = !m_nelem_plas;
    m_N = m_nelem_plas;

#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
    // check that "elem_plastic" and "elem_plastic" together span all elements
    xt::xtensor<size_t, 1> elem = xt::concatenate(xt::xtuple(m_elem_elas, m_elem_plas));
    FRICTIONQPOTFEM_ASSERT(xt::sort(elem) == xt::arange<size_t>(m_nelem));
    // check that all "iip" or in "dofs"
    FRICTIONQPOTFEM_ASSERT(xt::all(xt::isin(m_iip, m_dofs)));
#endif

    m_vector = GooseFEM::VectorPartitioned(m_conn, m_dofs, m_iip);

    m_quad = GooseFEM::Element::Quad4::Quadrature(m_vector.AsElement(m_coor));
    m_nip = m_quad.nip();

    m_u = m_vector.allocate_nodevec(0.0);
    m_v = m_vector.allocate_nodevec(0.0);
    m_a = m_vector.allocate_nodevec(0.0);
    m_v_n = m_vector.allocate_nodevec(0.0);
    m_a_n = m_vector.allocate_nodevec(0.0);

    m_ue = m_vector.allocate_elemvec(0.0);
    m_fe = m_vector.allocate_elemvec(0.0);

    m_fmaterial = m_vector.allocate_nodevec(0.0);
    m_fdamp = m_vector.allocate_nodevec(0.0);
    m_fint = m_vector.allocate_nodevec(0.0);
    m_fext = m_vector.allocate_nodevec(0.0);
    m_fres = m_vector.allocate_nodevec(0.0);

    m_Eps = m_quad.allocate_qtensor<2>(0.0);
    m_Sig = m_quad.allocate_qtensor<2>(0.0);

    m_M = GooseFEM::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);
    m_D = GooseFEM::MatrixDiagonal(m_conn, m_dofs);
    m_K = GooseFEM::MatrixPartitioned(m_conn, m_dofs, m_iip);

    m_material = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem, m_nip});
}

inline size_t System::N() const
{
    return m_nelem_plas;
}

inline std::string System::type() const
{
    return "FrictionQPotFEM.Generic2d.System";
}

inline void System::evalAllSet()
{
    m_allset = m_set_M && m_set_D && m_set_elas && m_set_plas && m_dt > 0.0;
}

template <class T>
inline void System::setMassMatrix(const T& val_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_M);
    FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

    GooseFEM::Element::Quad4::Quadrature nodalQuad(
        m_vector.AsElement(m_coor),
        GooseFEM::Element::Quad4::Nodal::xi(),
        GooseFEM::Element::Quad4::Nodal::w());

    xt::xtensor<double, 2> val_quad(std::array<size_t, 2>{m_nelem, nodalQuad.nip()});
    for (size_t q = 0; q < nodalQuad.nip(); ++q) {
        xt::view(val_quad, xt::all(), q) = val_elem;
    }

    m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    m_set_M = true;
    this->evalAllSet();
}

template <class T>
inline void System::setDampingMatrix(const T& val_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_D);
    FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

    GooseFEM::Element::Quad4::Quadrature nodalQuad(
        m_vector.AsElement(m_coor),
        GooseFEM::Element::Quad4::Nodal::xi(),
        GooseFEM::Element::Quad4::Nodal::w());

    xt::xtensor<double, 2> val_quad(std::array<size_t, 2>{m_nelem, nodalQuad.nip()});
    for (size_t q = 0; q < nodalQuad.nip(); ++q) {
        xt::view(val_quad, xt::all(), q) = val_elem;
    }

    m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    m_set_D = true;
    this->evalAllSet();
}

inline void System::initMaterial()
{
    if (!(m_set_elas && m_set_plas)) {
        return;
    }

    FRICTIONQPOTFEM_REQUIRE(
        xt::all(xt::not_equal(m_material.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));

    m_material.setStrain(m_Eps);

    m_K.assemble(m_quad.Int_gradN_dot_tensor4_dot_gradNT_dV(m_material.Tangent()));
}

inline void
System::setElastic(const xt::xtensor<double, 1>& K_elem, const xt::xtensor<double, 1>& G_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_elas || m_nelem_elas == 0);
    FRICTIONQPOTFEM_ASSERT(K_elem.size() == m_nelem_elas);
    FRICTIONQPOTFEM_ASSERT(G_elem.size() == m_nelem_elas);

    if (m_nelem_elas > 0) {
        xt::xtensor<size_t, 2> I(std::array<size_t, 2>{m_nelem, m_nip}, 0);
        xt::xtensor<size_t, 2> idx(std::array<size_t, 2>{m_nelem, m_nip}, 0);

        xt::view(I, xt::keep(m_elem_elas), xt::all()) = 1ul;
        xt::view(idx, xt::keep(m_elem_elas), xt::all()) =
            xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});

        m_material.setElastic(I, idx, K_elem, G_elem);
    }

    m_set_elas = true;
    this->evalAllSet();
    this->initMaterial();
}

inline void System::setPlastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem,
    const xt::xtensor<double, 2>& epsy_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_plas || m_nelem_plas == 0);
    FRICTIONQPOTFEM_ASSERT(K_elem.size() == m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(G_elem.size() == m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(epsy_elem.shape(0) == m_nelem_plas);

    if (m_nelem_plas > 0) {
        xt::xtensor<size_t, 2> I(std::array<size_t, 2>{m_nelem, m_nip}, 0);
        xt::xtensor<size_t, 2> idx(std::array<size_t, 2>{m_nelem, m_nip}, 0);

        xt::view(I, xt::keep(m_elem_plas), xt::all()) = 1ul;
        xt::view(idx, xt::keep(m_elem_plas), xt::all()) =
            xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});

        m_material.setCusp(I, idx, K_elem, G_elem, epsy_elem);
    }

    m_set_plas = true;
    this->evalAllSet();
    this->initMaterial();
}

inline void System::reset_epsy(const xt::xtensor<double, 2>& epsy_elem)
{
    FRICTIONQPOTFEM_ASSERT(m_set_plas);
    FRICTIONQPOTFEM_ASSERT(epsy_elem.shape(0) == m_nelem_plas);

    if (m_nelem_plas > 0) {
        xt::xtensor<size_t, 2> I(std::array<size_t, 2>{m_nelem, m_nip}, 0);
        xt::xtensor<size_t, 2> idx(std::array<size_t, 2>{m_nelem, m_nip}, 0);

        xt::view(I, xt::keep(m_elem_plas), xt::all()) = 1ul;
        xt::view(idx, xt::keep(m_elem_plas), xt::all()) =
            xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});

        m_material.reset_epsy(I, idx, epsy_elem);
    }
}

inline xt::xtensor<double, 2> System::epsy() const
{
    xt::xtensor<double, 2> ret;

    for (size_t e = 0; e < m_nelem_plas; ++e) {
        auto cusp = m_material.crefCusp({m_elem_plas(e), 0});
        auto val = cusp.epsy();
        if (e == 0) {
            ret.resize({m_nelem_plas, val.size()});
        }
        std::copy(val.cbegin(), val.cend(), &ret(e, 0));
    }

    return ret;
}

inline bool System::isHomogeneousElastic() const
{
    auto K = m_material.K();
    auto G = m_material.G();

    return xt::allclose(K, K.data()[0]) && xt::allclose(G, G.data()[0]);
}

inline void System::setT(double t)
{
    m_t = t;
}

inline void System::setDt(double dt)
{
    m_dt = dt;
    this->evalAllSet();
}

template <class T>
inline void System::setU(const T& u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, {m_nnode, m_ndim}));
    xt::noalias(m_u) = u;
    this->updated_u();
}

inline void System::updated_u()
{
    this->computeForceMaterial();
}

template <class T>
inline void System::setV(const T& v)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, {m_nnode, m_ndim}));
    xt::noalias(m_v) = v;
}

inline void System::updated_v()
{
    m_D.dot(m_v, m_fdamp);
}

template <class T>
inline void System::setA(const T& a)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(a, {m_nnode, m_ndim}));
    xt::noalias(m_a) = a;
}

template <class T>
inline void System::setFext(const T& fext)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(fext, {m_nnode, m_ndim}));
    xt::noalias(m_fext) = fext;
}

inline void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
}

inline auto System::elastic() const
{
    return m_elem_elas;
}

inline auto System::plastic() const
{
    return m_elem_plas;
}

inline auto System::conn() const
{
    return m_conn;
}

inline auto System::coor() const
{
    return m_coor;
}

inline auto System::dofs() const
{
    return m_dofs;
}

inline auto System::u() const
{
    return m_u;
}

inline auto System::v() const
{
    return m_v;
}

inline auto System::a() const
{
    return m_a;
}

inline auto& System::mass() const
{
    return m_M;
}

inline auto& System::damping() const
{
    return m_D;
}

inline auto System::fext() const
{
    return m_fext;
}

inline auto System::fint() const
{
    return m_fint;
}

inline auto System::fmaterial() const
{
    return m_fmaterial;
}

inline auto System::fdamp() const
{
    return m_fdamp;
}

inline double System::residual() const
{
    double r_fres = xt::norm_l2(m_fres)();
    double r_fext = xt::norm_l2(m_fext)();
    if (r_fext != 0.0) {
        return r_fres / r_fext;
    }
    return r_fres;
}

inline double System::t() const
{
    return m_t;
}

inline auto System::dV() const
{
    return m_quad.dV();
}

inline const GooseFEM::MatrixPartitioned& System::stiffness() const
{
    return m_K;
}

inline const GooseFEM::VectorPartitioned& System::vector() const
{
    return m_vector;
}

inline const GooseFEM::Element::Quad4::Quadrature& System::quad() const
{
    return m_quad;
}

inline const GMatElastoPlasticQPot::Cartesian2d::Array<2>& System::material() const
{
    return m_material;
}

inline xt::xtensor<double, 4> System::Sig()
{
    return m_Sig;
}

inline xt::xtensor<double, 4> System::Eps()
{
    return m_Eps;
}

inline xt::xtensor<double, 4> System::plastic_Sig() const
{
    return xt::view(m_Sig, xt::keep(m_elem_plas), xt::all(), xt::all(), xt::all());
}

inline xt::xtensor<double, 4> System::plastic_Eps() const
{
    return xt::view(m_Eps, xt::keep(m_elem_plas), xt::all(), xt::all(), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_Eps(size_t e_plastic, size_t q) const
{
    FRICTIONQPOTFEM_ASSERT(e_plastic < m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(q < m_nip);
    return xt::view(m_Eps, m_elem_plas(e_plastic), q, xt::all(), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldLeft() const
{
    return xt::view(m_material.CurrentYieldLeft(), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldRight() const
{
    return xt::view(m_material.CurrentYieldRight(), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldLeft(size_t offset) const
{
    return xt::view(m_material.CurrentYieldLeft(offset), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldRight(size_t offset) const
{
    return xt::view(m_material.CurrentYieldRight(offset), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<size_t, 2> System::plastic_CurrentIndex() const
{
    return xt::view(m_material.CurrentIndex(), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_Epsp() const
{
    return xt::view(m_material.Epsp(), xt::keep(m_elem_plas), xt::all());
}

template <class T>
inline xt::xtensor<int, 2> System::plastic_SignDeltaEpsd(const T& delta_u)
{
    FRICTIONQPOTFEM_WARNING_PYTHON("Deprecation: for now there is no use for this function");

    FRICTIONQPOTFEM_ASSERT(xt::has_shape(delta_u, m_u.shape()));
    auto u_0 = m_u;
    auto eps_0 = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    this->setU(m_u + delta_u);
    auto eps_pert = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    this->setU(u_0);
    xt::xtensor<int, 2> sign = xt::sign(eps_pert - eps_0);
    return sign;
}

inline bool System::boundcheck_right(size_t n) const
{
    return m_material.checkYieldBoundRight(n);
}

inline void System::computeStress()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);

    m_vector.asElement(m_u, m_ue);
    m_quad.symGradN_vector(m_ue, m_Eps);
    m_material.setStrain(m_Eps);
    m_material.stress(m_Sig);
}

inline void System::computeForceMaterial()
{
    this->computeStress();

    m_quad.int_gradN_dot_tensor2_dV(m_Sig, m_fe);
    m_vector.assembleNode(m_fe, m_fmaterial);
}

inline void System::computeInternalExternalResidualForce()
{
    xt::noalias(m_fint) = m_fmaterial + m_fdamp;
    m_vector.copy_p(m_fint, m_fext);
    xt::noalias(m_fres) = m_fext - m_fint;
}

template <class T>
inline double System::eventDriven_setDeltaU(const T& u, bool autoscale)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, m_u.shape()));
    m_pert_u = u;

    auto u0 = m_u;
    this->setU(u);
    m_pert_Epsd_plastic = GMatElastoPlasticQPot::Cartesian2d::Deviatoric(this->plastic_Eps());
    this->setU(u0);

    if (!autoscale) {
        return 1.0;
    }

    auto deps = xt::amax(GMatElastoPlasticQPot::Cartesian2d::Epsd(m_pert_Epsd_plastic))();
    auto d = xt::amin(xt::diff(this->epsy(), 1))();
    double c = 0.1 * d / deps;

    m_pert_u *= c;
    m_pert_Epsd_plastic *= c;

    return c;
}

inline auto System::eventDriven_deltaU() const
{
    return m_pert_u;
}

inline double System::eventDrivenStep(
    double deps_kick,
    bool kick,
    int direction,
    bool yield_element,
    bool iterative)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(m_pert_u, m_u.shape()));
    FRICTIONQPOTFEM_ASSERT(direction == 1 || direction == -1);

    auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    auto idx = this->plastic_CurrentIndex();
    auto epsy_l = this->plastic_CurrentYieldLeft();
    auto epsy_r = this->plastic_CurrentYieldRight();

    FRICTIONQPOTFEM_WIP(iterative || direction > 0 || !xt::any(xt::equal(idx, 0)));

    xt::xtensor<double, 2> target;

    if (direction > 0 && kick) { // direction > 0 && kick
        target = epsy_r + 0.5 * deps_kick;
    }
    else if (direction > 0) { // direction > 0 && !kick
        target = epsy_r - 0.5 * deps_kick;
    }
    else if (kick) { // direction < 0 && kick
        target = epsy_l - 0.5 * deps_kick;
    }
    else { // direction < 0 && !kick
        target = epsy_l + 0.5 * deps_kick;
    }

    auto Epsd_plastic = GMatElastoPlasticQPot::Cartesian2d::Deviatoric(this->plastic_Eps());
    auto scale = xt::empty_like(target);

    for (size_t e = 0; e < m_nelem_plas; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            double e_t = Epsd_plastic(e, q, 0, 0);
            double g_t = Epsd_plastic(e, q, 0, 1);
            double e_d = m_pert_Epsd_plastic(e, q, 0, 0);
            double g_d = m_pert_Epsd_plastic(e, q, 0, 1);
            double epsd_target = target(e, q);

            double a = e_d * e_d + g_d * g_d;
            double b = 2.0 * (e_t * e_d + g_t * g_d);
            double c = e_t * e_t + g_t * g_t - epsd_target * epsd_target;
            double D = std::sqrt(b * b - 4.0 * a * c);

            FRICTIONQPOTFEM_REQUIRE(b >= 0.0);
            scale(e, q) = (-b + D) / (2.0 * a);
        }
    }

    double ret;
    size_t e;
    size_t q;

    if (!iterative) {

        auto index = xt::unravel_index(xt::argmin(xt::abs(scale))(), scale.shape());
        e = index[0];
        q = index[1];

        if (kick && yield_element) {
            q = xt::argmax(xt::view(xt::abs(scale), e, xt::all()))();
        }

        ret = scale(e, q);

        if ((direction > 0 && ret < 0) || (direction < 0 && ret > 0)) {
            if (!kick) {
                return 0.0;
            }
            else {
                FRICTIONQPOTFEM_REQUIRE((direction > 0 && ret > 0) || (direction < 0 && ret < 0));
            }
        }
    }

    else {

        double dir = static_cast<double>(direction);
        auto steps = xt::sort(xt::ravel(xt::eval(xt::abs(scale))));
        auto u_n = this->u();
        auto jdx = xt::cast<long>(this->plastic_CurrentIndex());
        size_t i;
        long nip = static_cast<long>(m_nip);

        // find first step that:
        // if (!kick || (kick && !yield_element)): is plastic for the first time
        // if (kick && yield_element): yields the element for the first time

        for (i = 0; i < steps.size(); ++i) {
            this->setU(u_n + dir * steps(i) * m_pert_u);
            auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
            auto S = xt::abs(jdx_new - jdx);
            if (!yield_element || !kick) {
                if (xt::any(S > 0)) {
                    break;
                }
            }
            else {
                if (xt::any(xt::equal(xt::sum(S > 0, 1), nip))) {
                    break;
                }
            }
        }

        // iterate to acceptable step

        double right = steps(i);
        double left = 0.0;
        ret = right;

        if (i > 0) {
            left = steps(i - 1);
        }

        // iterate to actual step

        for (size_t iiter = 0; iiter < 1100; ++iiter) {

            ret = 0.5 * (right + left);
            this->setU(u_n + dir * ret * m_pert_u);
            auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
            auto S = xt::abs(jdx_new - jdx);

            if (!kick) {
                if (xt::any(S > 0)) {
                    right = ret;
                }
                else {
                    left = ret;
                }
            }
            else if (yield_element) {
                if (xt::any(xt::equal(xt::sum(S > 0, 1), nip))) {
                    right = ret;
                }
                else {
                    left = ret;
                }
            }
            else {
                if (xt::any(S > 0)) {
                    right = ret;
                }
                else {
                    left = ret;
                }
            }

            if ((right - left) / left < 1e-5) {
                break;
            }
            FRICTIONQPOTFEM_REQUIRE(iiter < 1000);
        }

        // final assertion: make sure that "left" and "right" are still bounds

        {
            this->setU(u_n + dir * left * m_pert_u);
            auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
            auto S = xt::abs(jdx_new - jdx);

            FRICTIONQPOTFEM_REQUIRE(kick || xt::all(xt::equal(S, 0)));
            if (kick && yield_element) {
                FRICTIONQPOTFEM_REQUIRE(!xt::any(xt::equal(xt::sum(S > 0, 1), nip)));
            }
            else if (kick) {
                FRICTIONQPOTFEM_REQUIRE(xt::all(xt::equal(S, 0)));
            }
        }
        {
            this->setU(u_n + dir * right * m_pert_u);
            auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
            auto S = xt::abs(jdx_new - jdx);
            FRICTIONQPOTFEM_REQUIRE(!xt::all(xt::equal(S, 0)));

            FRICTIONQPOTFEM_REQUIRE(kick || !xt::all(xt::equal(S, 0)));
            if (kick && yield_element) {
                FRICTIONQPOTFEM_REQUIRE(xt::any(xt::equal(xt::sum(S > 0, 1), nip)));
            }
            else if (kick) {
                FRICTIONQPOTFEM_REQUIRE(xt::any(S > 0));
            }
        }

        // "output"

        if (!kick) {
            ret = dir * left;
        }
        else {
            ret = dir * right;
        }
        this->setU(u_n);
        FRICTIONQPOTFEM_REQUIRE((direction > 0 && ret >= 0) || (direction < 0 && ret <= 0));
    }

    this->setU(m_u + ret * m_pert_u);

    auto idx_new = this->plastic_CurrentIndex();
    FRICTIONQPOTFEM_REQUIRE(kick || xt::all(xt::equal(idx, idx_new)));
    FRICTIONQPOTFEM_REQUIRE(!kick || xt::any(xt::not_equal(idx, idx_new)));

    if (!iterative) {
        auto eps_new = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps())(e, q);
        auto eps_target = target(e, q);
        FRICTIONQPOTFEM_REQUIRE(xt::allclose(eps_new, eps_target));
    }

    return ret;
}

inline void System::timeStep()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);

    // history

    m_t += m_dt;
    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    // new displacement

    xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
    this->updated_u();

    // estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
    this->updated_v();

    // compute residual force & solve

    this->computeInternalExternalResidualForce();
    m_M.solve(m_fres, m_a);

    // re-estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    // compute residual force & solve

    this->computeInternalExternalResidualForce();
    m_M.solve(m_fres, m_a);

    // new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    // compute residual force & solve

    this->computeInternalExternalResidualForce();
    m_M.solve(m_fres, m_a);
}

inline void System::timeSteps(size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        this->timeStep();
    }
}

inline size_t System::timeSteps_residualcheck(size_t n, double tol, size_t niter_tol)
{
    FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (size_t i = 0; i < n; ++i) {

        this->timeStep();

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return 0;
        }
    }

    return n;
}

inline size_t System::timeSteps_boundcheck(size_t n, size_t nmargin)
{
    if (!this->boundcheck_right(nmargin)) {
        return 0;
    }

    for (size_t i = 0; i < n; ++i) {
        this->timeStep();

        if (!this->boundcheck_right(nmargin)) {
            return 0;
        }
    }

    return n;
}

template <class T>
inline void System::flowSteps(size_t n, const T& v)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, m_u.shape()));

    for (size_t i = 0; i < n; ++i) {
        m_u += v * m_dt;
        this->timeStep();
    }
}

template <class T>
inline size_t System::flowSteps_boundcheck(size_t n, const T& v, size_t nmargin)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, m_u.shape()));

    if (!this->boundcheck_right(nmargin)) {
        return 0;
    }

    for (size_t i = 0; i < n; ++i) {
        m_u += v * m_dt;
        this->timeStep();

        if (!this->boundcheck_right(nmargin)) {
            return 0;
        }
    }

    return n;
}

inline size_t System::timeStepsUntilEvent(double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
    size_t iiter;
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    auto idx_n = this->plastic_CurrentIndex();

    for (iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        auto idx = this->plastic_CurrentIndex();

        if (xt::any(xt::not_equal(idx, idx_n))) {
            return iiter;
        }

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return 0;
        }
    }

    return iiter;
}

inline size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    std::cout << "residuals = " << xt::adapt(residuals.get()) << std::endl;
    bool converged = false;
    FRICTIONQPOTFEM_REQUIRE(converged == true);
    return 0; // irrelevant, the code never goes here
}

inline size_t
System::minimise_boundcheck(size_t nmargin, double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }

        if (!this->boundcheck_right(nmargin)) {
            return 0;
        }
    }

    std::cout << "residuals = " << xt::adapt(residuals.get()) << std::endl;
    bool converged = false;
    FRICTIONQPOTFEM_REQUIRE(converged == true);
    return 0; // irrelevant, the code never goes here
}

template <class T>
inline size_t System::minimise_truncate(
    const T& idx_n,
    size_t A_truncate,
    size_t S_truncate,
    double tol,
    size_t niter_tol,
    size_t max_iter)
{
    FRICTIONQPOTFEM_REQUIRE(xt::has_shape(idx_n, std::array<size_t, 1>{m_N}));
    FRICTIONQPOTFEM_REQUIRE(S_truncate == 0);
    FRICTIONQPOTFEM_REQUIRE(A_truncate > 0);
    FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }

        xt::xtensor<size_t, 1> idx = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
        if (xt::sum(xt::not_equal(idx_n, idx))() >= A_truncate) {
            return 0;
        }
    }

    std::cout << "residuals = " << xt::adapt(residuals.get()) << std::endl;
    bool converged = false;
    FRICTIONQPOTFEM_REQUIRE(converged == true);
    return 0; // irrelevant, the code never goes here
}

inline size_t System::minimise_truncate(
    size_t A_truncate,
    size_t S_truncate,
    double tol,
    size_t niter_tol,
    size_t max_iter)
{
    xt::xtensor<size_t, 1> idx_n = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
    return this->minimise_truncate(idx_n, A_truncate, S_truncate, tol, niter_tol, max_iter);
}

inline xt::xtensor<double, 2> System::affineSimpleShear(double delta_gamma) const
{
    xt::xtensor<double, 2> ret = xt::zeros_like(m_u);

    for (size_t n = 0; n < m_nnode; ++n) {
        ret(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - m_coor(0, 1));
    }

    return ret;
}

inline xt::xtensor<double, 2> System::affineSimpleShearCentered(double delta_gamma) const
{
    xt::xtensor<double, 2> ret = xt::zeros_like(m_u);
    size_t ll = m_conn(m_elem_plas(0), 0);
    size_t ul = m_conn(m_elem_plas(0), 3);
    double y0 = (m_coor(ul, 1) + m_coor(ll, 1)) / 2.0;

    for (size_t n = 0; n < m_nnode; ++n) {
        ret(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - y0);
    }

    return ret;
}

template <class C, class E, class L>
inline HybridSystem::HybridSystem(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initHybridSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

template <class C, class E, class L>
inline void HybridSystem::initHybridSystem(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);

    m_conn_elas = xt::view(m_conn, xt::keep(m_elem_elas), xt::all());
    m_conn_plas = xt::view(m_conn, xt::keep(m_elem_plas), xt::all());

    m_vector_elas = GooseFEM::VectorPartitioned(m_conn_elas, m_dofs, m_iip);
    m_vector_plas = GooseFEM::VectorPartitioned(m_conn_plas, m_dofs, m_iip);

    m_quad_elas = GooseFEM::Element::Quad4::Quadrature(m_vector_elas.AsElement(m_coor));
    m_quad_plas = GooseFEM::Element::Quad4::Quadrature(m_vector_plas.AsElement(m_coor));

    m_ue_plas = m_vector_plas.allocate_elemvec(0.0);
    m_fe_plas = m_vector_plas.allocate_elemvec(0.0);

    m_felas = m_vector.allocate_nodevec(0.0);
    m_fplas = m_vector.allocate_nodevec(0.0);

    m_Eps_elas = m_quad_elas.allocate_qtensor<2>(0.0);
    m_Sig_elas = m_quad_elas.allocate_qtensor<2>(0.0);
    m_Eps_plas = m_quad_plas.allocate_qtensor<2>(0.0);
    m_Sig_plas = m_quad_plas.allocate_qtensor<2>(0.0);

    m_material_elas = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem_elas, m_nip});
    m_material_plas = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem_plas, m_nip});

    m_K_elas = GooseFEM::Matrix(m_conn_elas, m_dofs);
}

inline std::string HybridSystem::type() const
{
    return "FrictionQPotFEM.Generic2d.HybridSystem";
}

inline void
HybridSystem::setElastic(const xt::xtensor<double, 1>& K_elem, const xt::xtensor<double, 1>& G_elem)
{
    System::setElastic(K_elem, G_elem);

    if (m_nelem_elas == 0) {
        return;
    }

    xt::xtensor<size_t, 2> I(std::array<size_t, 2>{m_nelem_elas, m_nip}, 1);
    xt::xtensor<size_t, 2> idx(std::array<size_t, 2>{m_nelem_elas, m_nip}, 0);

    xt::view(idx, xt::range(0, m_nelem_elas), xt::all()) =
        xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});

    m_material_elas.setElastic(I, idx, K_elem, G_elem);
    m_material_elas.setStrain(m_Eps_elas);

    FRICTIONQPOTFEM_REQUIRE(xt::all(
        xt::not_equal(m_material_elas.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));

    m_K_elas.assemble(m_quad_elas.Int_gradN_dot_tensor4_dot_gradNT_dV(m_material_elas.Tangent()));
}

inline void HybridSystem::setPlastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem,
    const xt::xtensor<double, 2>& epsy_elem)
{
    System::setPlastic(K_elem, G_elem, epsy_elem);

    if (m_nelem_plas == 0) {
        return;
    }

    xt::xtensor<size_t, 2> I(std::array<size_t, 2>{m_nelem_plas, m_nip}, 1);
    xt::xtensor<size_t, 2> idx(std::array<size_t, 2>{m_nelem_plas, m_nip}, 0);

    xt::view(idx, xt::range(0, m_nelem_plas), xt::all()) =
        xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});

    m_material_plas.setCusp(I, idx, K_elem, G_elem, epsy_elem);
    m_material_plas.setStrain(m_Eps_plas);

    FRICTIONQPOTFEM_REQUIRE(xt::all(
        xt::not_equal(m_material_plas.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));
}

inline void HybridSystem::reset_epsy(const xt::xtensor<double, 2>& epsy_elem)
{
    System::reset_epsy(epsy_elem);

    if (m_nelem_plas == 0) {
        return;
    }

    xt::xtensor<size_t, 2> I(std::array<size_t, 2>{m_nelem_plas, m_nip}, 1);
    xt::xtensor<size_t, 2> idx(std::array<size_t, 2>{m_nelem_plas, m_nip}, 0);

    xt::view(idx, xt::range(0, m_nelem_plas), xt::all()) =
        xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});

    m_material_plas.reset_epsy(I, idx, epsy_elem);
}

inline const GMatElastoPlasticQPot::Cartesian2d::Array<2>& HybridSystem::material_elastic() const
{
    return m_material_elas;
}

inline const GMatElastoPlasticQPot::Cartesian2d::Array<2>& HybridSystem::material_plastic() const
{
    return m_material_plas;
}

inline void HybridSystem::evalSystem()
{
    if (!m_eval_full) {
        return;
    }
    this->computeStress();
    m_eval_full = false;
}

inline xt::xtensor<double, 4> HybridSystem::Sig()
{
    this->evalSystem();
    return m_Sig;
}

inline xt::xtensor<double, 4> HybridSystem::Eps()
{
    this->evalSystem();
    return m_Eps;
}

inline xt::xtensor<double, 4> HybridSystem::plastic_Sig() const
{
    return m_Sig_plas;
}

inline xt::xtensor<double, 4> HybridSystem::plastic_Eps() const
{
    return m_Eps_plas;
}

inline xt::xtensor<double, 2> HybridSystem::plastic_Eps(size_t e_plastic, size_t q) const
{
    FRICTIONQPOTFEM_ASSERT(e_plastic < m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(q < m_nip);
    return xt::view(m_Eps_plas, e_plastic, q, xt::all(), xt::all());
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldLeft() const
{
    return m_material_plas.CurrentYieldLeft();
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldRight() const
{
    return m_material_plas.CurrentYieldRight();
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldLeft(size_t offset) const
{
    return m_material_plas.CurrentYieldLeft(offset);
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldRight(size_t offset) const
{
    return m_material_plas.CurrentYieldRight(offset);
}

inline xt::xtensor<size_t, 2> HybridSystem::plastic_CurrentIndex() const
{
    return m_material_plas.CurrentIndex();
}

inline xt::xtensor<double, 2> HybridSystem::plastic_Epsp() const
{
    return m_material_plas.Epsp();
}

inline bool HybridSystem::boundcheck_right(size_t n) const
{
    return m_material_plas.checkYieldBoundRight(n);
}

inline void HybridSystem::computeForceMaterial()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);
    m_eval_full = true;

    m_vector_plas.asElement(m_u, m_ue_plas);
    m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
    m_material_plas.setStrain(m_Eps_plas);
    m_material_plas.stress(m_Sig_plas);
    m_quad_plas.int_gradN_dot_tensor2_dV(m_Sig_plas, m_fe_plas);
    m_vector_plas.assembleNode(m_fe_plas, m_fplas);

    m_K_elas.dot(m_u, m_felas);

    xt::noalias(m_fmaterial) = m_felas + m_fplas;
}

} // namespace Generic2d
} // namespace FrictionQPotFEM

#endif

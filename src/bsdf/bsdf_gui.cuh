#pragma once
#include "bsdf_structs.cuh"
#include "imgui.h"

static bool draw_bsdf(Lambertian& m) {
    return ImGui::ColorEdit3("color", &m.color.x);
}
static bool draw_bsdf(Conductor& m) {
    bool r = false;
    r |= ImGui::DragFloat3("eta", &m.eta.x, 0.01f, 0.f, 10.f);
    r |= ImGui::DragFloat3("k", &m.k.x, 0.01f, 0.f, 10.f);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("anisotropy", &m.anisotropy, -1.f, 1.f);
    return r;
}
static bool draw_bsdf(Dielectric& m) {
    bool r = false;
    r |= ImGui::SliderFloat("ior", &m.ior, 1.f, 3.f);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    return r;
}
static bool draw_bsdf(DisneyDiffuse& m) {
    bool r = false;
    r |= ImGui::ColorEdit3("color", &m.color.x);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("subsurface", &m.subsurface, 0.f, 1.f);
    return r;
}
static bool draw_bsdf(DisneyMetal& m) {
    bool r = false;
    r |= ImGui::ColorEdit3("color", &m.color.x);
    r |= ImGui::ColorEdit3("edge_tint", &m.edge_tint.x);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("anisotropic", &m.anisotropic, 0.f, 1.f);
    return r;
}
static bool draw_bsdf(DisneyClearcoat& m) {
    return ImGui::SliderFloat("gloss", &m.gloss, 0.f, 1.f);
}
static bool draw_bsdf(DisneyGlass& m) {
    bool r = false;
    r |= ImGui::ColorEdit3("color", &m.color.x);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("anisotropic", &m.anisotropic, 0.f, 1.f);
    r |= ImGui::SliderFloat("ior", &m.ior, 1.f, 3.f);
    return r;
}
static bool draw_bsdf(DisneySheen& m) {
    bool r = false;
    r |= ImGui::ColorEdit3("color", &m.color.x);
    r |= ImGui::SliderFloat("sheen_tint", &m.sheen_tint, 0.f, 1.f);
    return r;
}
static bool draw_bsdf(DisneyBsdf& m) {
    bool r = false;
    r |= ImGui::ColorEdit3("color", &m.color.x);
    r |= ImGui::SliderFloat("specular_transmission", &m.specular_transmission, 0.f, 1.f);
    r |= ImGui::SliderFloat("metallic", &m.metallic, 0.f, 1.f);
    r |= ImGui::ColorEdit3("edge_tint", &m.edge_tint.x);
    r |= ImGui::SliderFloat("subsurface", &m.subsurface, 0.f, 1.f);
    r |= ImGui::SliderFloat("specular", &m.specular, 0.f, 1.f);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("specular_tint", &m.specular_tint, 0.f, 1.f);
    r |= ImGui::SliderFloat("anisotropic", &m.anisotropic, 0.f, 1.f);
    r |= ImGui::SliderFloat("sheen", &m.sheen, 0.f, 1.f);
    r |= ImGui::SliderFloat("sheen_tint", &m.sheen_tint, 0.f, 1.f);
    r |= ImGui::SliderFloat("clearcoat", &m.clearcoat, 0.f, 1.f);
    r |= ImGui::SliderFloat("clearcoat_gloss", &m.clearcoat_gloss, 0.f, 1.f);
    r |= ImGui::SliderFloat("ior", &m.ior, 1.f, 3.f);
    return r;
}

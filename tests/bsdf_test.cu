// Checks scatterPbr and evalPbr against a brute-force integral of the analytic BSDF, then fuzzes them; nonzero exit on a failure.
// The README's "Checking the BSDF" table comes from this output.
#include "render/bsdf.cu"

#include <cstdio>
#include <cmath>

// Combined standard errors two estimates may differ by; the seeds are fixed, so results repeat.
static const double MAX_SIGMA = 5.0;

struct Estimate
{
    glm::dvec3 sum = glm::dvec3(0.0);
    glm::dvec3 sumSq = glm::dvec3(0.0);

    void add(glm::dvec3 x)
    {
        sum += x;
        sumSq += x * x;
    }
    glm::dvec3 mean(double n) const { return sum / n; }
    glm::dvec3 standardError(double n) const
    {
        const glm::dvec3 m = sum / n;
        return glm::sqrt(glm::max(sumSq / n - m * m, glm::dvec3(0.0)) / n);
    }
};

static int checks = 0;
static int failures = 0;

static bool check(bool ok)
{
    ++checks;
    failures += ok ? 0 : 1;
    return ok;
}

static float D(float cosH, float a)
{
    if (cosH <= 0) return 0;
    const float a2 = a * a;
    const float d = cosH * cosH * (a2 - 1) + 1;
    return a2 / (PI * d * d);
}
static float lambda(float c, float a) { return ggxLambda(fabsf(c), a); }
static float G2(float co, float ci, float a) { return 1.0f / (1.0f + lambda(co, a) + lambda(ci, a)); }

static glm::vec3 schlickRef(glm::vec3 f0, float c) { return schlickFresnel(f0, c); }

struct Case
{
    const char* name;
    Material m;
    PbrSurface s;
};

// Analytic f * |cos wi| with normal (0, 0, 1) and wo outside.
static glm::vec3 fcos(const Case& c, glm::vec3 wo, glm::vec3 wi)
{
    const glm::vec3 n(0, 0, 1);
    const float co = wo.z;
    const float ci = wi.z;
    const float a = c.s.roughness * c.s.roughness;
    glm::vec3 coat(0.0f);
    float coatW = 0;
    if (c.m.clearcoat > 0)
    {
        coatW = c.m.clearcoat * schlickRef(glm::vec3(0.04f), co).x;
        const float ca = c.m.clearcoatRoughness * c.m.clearcoatRoughness;
        if (ci > 0)
        {
            const glm::vec3 h = glm::normalize(wo + wi);
            coat = glm::vec3(D(h.z, ca) * G2(co, ci, ca) / (4 * co * ci) * ci);
        }
    }
    glm::vec3 base(0.0f);
    const float metal = c.s.metallic;
    const glm::vec3 f0 = glm::min(iorF0(c.m.ior) * c.m.specularColorFactor, glm::vec3(1.0f));
    const float w = c.m.specularFactor;
    if (ci > 0)
    {
        const glm::vec3 h = glm::normalize(wo + wi);
        const float spec = D(h.z, a) * G2(co, ci, a) / (4 * co * ci) * ci;
        base += metal * schlickRef(c.s.baseColor, glm::dot(wo, h)) * spec;
        const glm::vec3 F = w * schlickRef(f0, glm::dot(wo, h));
        const float maxF = maxComponent(F);
        base += (1 - metal) * (1 - c.s.transmission) * (F * spec + (1 - maxF) * c.s.baseColor / PI * ci);
        base += (1 - metal) * c.s.transmission * F * spec;
    }
    else if (c.s.transmission > 0)
    {
        // PBRT-v4 dielectric BTDF without the 1/eta^2 radiance factor
        const float etap = c.m.ior;  // n_t / n_i, wo outside
        glm::vec3 wm = glm::normalize(wi * etap + wo);
        if (wm.z < 0) wm = -wm;
        if (glm::dot(wm, wi) * ci > 0 && glm::dot(wm, wo) * co > 0)
        {
            const float denom = (glm::dot(wi, wm) + glm::dot(wo, wm) / etap);
            const float F = w * schlickRef(f0, glm::dot(wo, wm)).x;  // white specular color in the glass cases
            const float ft = (1 - F) * D(wm.z, a) * G2(co, ci, a)
                           * fabsf(glm::dot(wi, wm) * glm::dot(wo, wm) / (denom * denom * ci * co));
            base += (1 - metal) * c.s.transmission * ft * fabsf(ci) * c.s.baseColor;
        }
    }
    return (1 - coatW) * base + coatW * coat;
}

int main()
{
    std::vector<Case> cases;
    auto add = [&](const char* name, float metallic, float rough, float trans, glm::vec3 base, float spec, float cc, float ccr) {
        Case c{};
        c.name = name;
        c.m.ior = 1.5f;
        c.m.specularFactor = spec;
        c.m.specularColorFactor = glm::vec3(1.0f);
        c.m.clearcoat = cc;
        c.m.clearcoatRoughness = ccr;
        c.s.baseColor = base;
        c.s.metallic = metallic;
        c.s.roughness = rough;
        c.s.transmission = trans;
        c.s.normal = c.s.coatNormal = glm::vec3(0, 0, 1);
        cases.push_back(c);
    };
    for (float r : { 0.3f, 0.6f, 1.0f })
    {
        add("metal", 1, r, 0, glm::vec3(1.0f, 0.7f, 0.3f), 1, 0, 0);
        add("plastic", 0, r, 0, glm::vec3(0.5f, 0.8f, 0.2f), 1, 0, 0);
        add("plastic spec0", 0, r, 0, glm::vec3(0.5f, 0.8f, 0.2f), 0, 0, 0);
        add("glass", 0, r, 1, glm::vec3(1.0f), 1, 0, 0);
        add("mixed", 0.4f, r, 0.3f, glm::vec3(0.9f, 0.6f, 0.4f), 0.7f, 0, 0);
        add("coated plastic", 0, r, 0, glm::vec3(0.5f, 0.1f, 0.1f), 1, 1.0f, 0.3f);
    }

    const int N = 1 << 21;
    printf("%-15s %5s %6s  %-26s %-26s %5s\n", "case", "rough", "theta", "sampled albedo", "reference albedo", "z");
    for (const Case& c : cases)
    {
        for (float theta : { 0.1f, 0.8f, 1.3f })
        {
            const glm::vec3 wo(sinf(theta), 0, cosf(theta));
            thrust::default_random_engine rng(1234);
            thrust::uniform_real_distribution<float> u01(0, 1);
            Estimate sampled;
            for (int i = 0; i < N; ++i)
            {
                PathSegment p{};
                p.ray.direction = -wo;
                p.color = glm::vec3(1.0f);
                if (scatterPbr(p, glm::vec3(0.0f), glm::vec3(0, 0, 1), true, c.m, c.s, rng))
                {
                    sampled.add(glm::dvec3(p.color));
                }
            }
            Estimate ref;
            for (int i = 0; i < N * 4; ++i)
            {
                const float z = 1 - 2 * u01(rng);
                const float r = sqrtf(fmaxf(0, 1 - z * z));
                const float phi = TWO_PI * u01(rng);
                const glm::vec3 wi(r * cosf(phi), r * sinf(phi), z);
                ref.add(glm::dvec3(fcos(c, wo, wi)) * (4.0 * PI));
            }
            const glm::dvec3 sampledMean = sampled.mean(N);
            const glm::dvec3 refMean = ref.mean(N * 4.0);
            const glm::dvec3 sampledSe = sampled.standardError(N);
            const glm::dvec3 refSe = ref.standardError(N * 4.0);
            const glm::dvec3 se = glm::sqrt(sampledSe * sampledSe + refSe * refSe);
            double zMax = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                // A channel neither estimate has any variance in must match exactly.
                const double diff = fabs(sampledMean[k] - refMean[k]);
                zMax = fmax(zMax, se[k] > 0.0 ? diff / se[k] : (diff > 1e-12 ? INFINITY : 0.0));
            }
            const bool ok = check(zMax <= MAX_SIGMA);
            printf("%-15s %5.2f %6.2f  %.4f %.4f %.4f    %.4f %.4f %.4f    %5.2f%s\n", c.name, c.s.roughness, theta,
                sampledMean.x, sampledMean.y, sampledMean.z, refMean.x, refMean.y, refMean.z, zMax, ok ? "" : "  FAIL");
        }
    }

    // evalPbr: f * cos against fcos (front side only), pdf against scatterPbr's samples from both sides.
    {
        const int M = 1 << 16;
        const int N2 = 1 << 20;
        printf("\n%-15s %5s %6s %7s  %9s  %-15s %5s  %-26s %5s\n", "case", "rough", "theta", "side", "max error",
            "pdf int / kept", "z", "mean f cos / pdf", "z");
        for (const Case& c : cases)
        {
            for (bool outside : { true, false })
            {
                if (!outside && c.s.transmission == 0.0f)
                {
                    continue;
                }
                for (float theta : { 0.1f, 0.8f, 1.3f })
                {
                    const glm::vec3 n(0, 0, 1);
                    const glm::vec3 wo(sinf(theta), 0, cosf(theta));
                    thrust::default_random_engine rng(4321);
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    auto uniformSphere = [&]() {
                        const float z = 1 - 2 * u01(rng);
                        const float r = sqrtf(fmaxf(0, 1 - z * z));
                        const float phi = TWO_PI * u01(rng);
                        return glm::vec3(r * cosf(phi), r * sinf(phi), z);
                    };

                    // Relative error, with a 1e-3 floor for near-zero values
                    double maxError = 0.0;
                    if (outside)
                    {
                        for (int i = 0; i < M; ++i)
                        {
                            const glm::vec3 wi = uniformSphere();
                            const glm::vec3 e = evalPbr(wo, wi, n, true, c.m, c.s).fCos;
                            const glm::vec3 r = fcos(c, wo, wi);
                            maxError = fmax(maxError, maxComponent(glm::abs(e - r)) / (maxComponent(r) + 1e-3));
                        }
                    }

                    Estimate pdfIntegral;
                    for (int i = 0; i < N2; ++i)
                    {
                        pdfIntegral.add(glm::dvec3(evalPbr(wo, uniformSphere(), n, outside, c.m, c.s).pdf * 4.0 * PI));
                    }
                    Estimate weights;
                    Estimate ratios;
                    int kept = 0;
                    int pdfMismatches = 0;
                    for (int i = 0; i < N2; ++i)
                    {
                        PathSegment p{};
                        p.ray.direction = -wo;
                        p.color = glm::vec3(1.0f);
                        if (scatterPbr(p, glm::vec3(0.0f), n, outside, c.m, c.s, rng))
                        {
                            ++kept;
                            weights.add(glm::dvec3(p.color));
                            const BsdfEval e = evalPbr(wo, p.ray.direction, n, outside, c.m, c.s);
                            if (e.pdf > 0.0f)
                            {
                                ratios.add(glm::dvec3(e.fCos / e.pdf));
                            }
                            pdfMismatches += p.pdf != e.pdf;
                        }
                    }

                    const double keptShare = kept / (double)N2;
                    const double keptSe = sqrt(keptShare * (1.0 - keptShare) / N2);
                    const double pdfMean = pdfIntegral.mean(N2).x;
                    const double pdfSe = pdfIntegral.standardError(N2).x;
                    const double zPdf = fabs(pdfMean - keptShare) / sqrt(pdfSe * pdfSe + keptSe * keptSe);
                    const glm::dvec3 ratioMean = ratios.mean(N2);
                    const glm::dvec3 weightMean = weights.mean(N2);
                    const glm::dvec3 ratioSe = ratios.standardError(N2);
                    const glm::dvec3 weightSe = weights.standardError(N2);
                    // Pure Lambert has no variance, so a zero error still allows for float rounding.
                    double zRatio = 0.0;
                    for (int k = 0; k < 3; ++k)
                    {
                        const double se = sqrt(ratioSe[k] * ratioSe[k] + weightSe[k] * weightSe[k]);
                        const double diff = fabs(ratioMean[k] - weightMean[k]);
                        zRatio = fmax(zRatio, se > 0.0 ? diff / se : (diff > 1e-5 * weightMean[k] ? INFINITY : 0.0));
                    }
                    const bool ok = check(maxError <= 2e-3 && zPdf <= MAX_SIGMA && zRatio <= MAX_SIGMA
                        && pdfMismatches == 0);
                    printf("%-15s %5.2f %6.2f %7s  %9.2e  %.4f %.4f  %5.2f  %.4f %.4f %.4f  %5.2f%s\n", c.name,
                        c.s.roughness, theta, outside ? "front" : "inside", maxError, pdfMean, keptShare, zPdf,
                        ratioMean.x, ratioMean.y, ratioMean.z, zRatio, ok ? "" : "  FAIL");
                    if (pdfMismatches > 0)
                    {
                        printf("  %d samples whose path.pdf differs from evalPbr's pdf\n", pdfMismatches);
                    }
                }
            }
        }
        printf("\n");
    }

    {
        Case c = cases[0];
        c.s.roughness = 0;
        thrust::default_random_engine rng(7);
        PathSegment p{};
        const glm::vec3 wo = glm::normalize(glm::vec3(0.5f, 0.2f, 0.8f));
        p.ray.direction = -wo;
        p.color = glm::vec3(1);
        p.pdf = -1;
        const bool scattered = scatterPbr(p, glm::vec3(0), glm::vec3(0, 0, 1), true, c.m, c.s, rng);
        const glm::vec3 F = schlickFresnel(c.s.baseColor, wo.z);
        const glm::vec3 mirror(-wo.x, -wo.y, wo.z);
        // A smooth lobe has pdf 0, and evalPbr gives nothing even at its exact direction.
        const BsdfEval e = evalPbr(wo, mirror, glm::vec3(0, 0, 1), true, c.m, c.s);
        const bool ok = check(scattered && glm::length(p.color - F) < 1e-6f && glm::length(p.ray.direction - mirror) < 1e-6f
            && p.pdf == 0.0f && e.pdf == 0.0f && e.fCos == glm::vec3(0.0f));
        printf("smooth metal: weight %.4f %.4f %.4f expected %.4f %.4f %.4f, dir %.4f %.4f %.4f, pdf %g, eval %g%s\n",
            p.color.x, p.color.y, p.color.z, F.x, F.y, F.z, p.ray.direction.x, p.ray.direction.y, p.ray.direction.z, p.pdf,
            e.pdf, ok ? "" : "  FAIL");
    }
    {
        // smooth glass from inside at a TIR angle and at a transmitting angle
        Case c = cases[3];
        c.s.roughness = 0;
        for (float theta : { 0.3f, 0.9f })
        {
            thrust::default_random_engine rng(11);
            glm::dvec3 sum(0);
            int refl = 0;
            const int M = 100000;
            for (int i = 0; i < M; ++i)
            {
                PathSegment p{};
                const glm::vec3 wo(sinf(theta), 0, cosf(theta));
                p.ray.direction = -wo;
                p.color = glm::vec3(1);
                if (scatterPbr(p, glm::vec3(0), glm::vec3(0, 0, 1), false, c.m, c.s, rng))
                {
                    sum += glm::dvec3(p.color);
                    refl += p.ray.direction.z > 0;
                }
            }
            // White glass weighs exactly 1; the reflected share is Schlick at the transmitted angle, or 1 past TIR.
            const float sin2T = 1.5f * 1.5f * sinf(theta) * sinf(theta);
            const double expected = sin2T >= 1.0f ? 1.0 : schlickFresnel(glm::vec3(iorF0(1.5f)), sqrtf(1.0f - sin2T)).x;
            const double reflected = refl / (double)M;
            const double se = sqrt(expected * (1.0 - expected) / M);
            const bool ok = check(sum.x == M && fabs(reflected - expected) <= MAX_SIGMA * se);
            printf("smooth glass inside, theta %.2f: albedo %.4f, reflected %.4f expected %.4f (TIR above %.3f rad)%s\n",
                theta, sum.x / M, reflected, expected, asinf(1 / 1.5f), ok ? "" : "  FAIL");
        }
    }

    {
        thrust::default_random_engine rng(99);
        thrust::uniform_real_distribution<float> u01(0, 1);
        int bad = 0, big = 0, bigTypical = 0;
        double maxWeight = 0, maxTypical = 0;
        for (int i = 0; i < 2000000; ++i)
        {
            Case c = cases[0];
            c.m.ior = u01(rng) < 0.05f ? 0.0f : 1.0f + u01(rng);
            c.m.specularFactor = u01(rng);
            c.m.specularColorFactor = glm::vec3(u01(rng), u01(rng), u01(rng)) * 2.0f;
            c.m.clearcoat = u01(rng) < 0.5f ? 0 : u01(rng);
            c.m.clearcoatRoughness = u01(rng);
            c.s.baseColor = glm::vec3(u01(rng), u01(rng), u01(rng));
            c.s.metallic = u01(rng) < 0.3f ? 0 : u01(rng);
            c.s.roughness = u01(rng) < 0.2f ? 0 : u01(rng);
            c.s.transmission = u01(rng) < 0.5f ? 0 : u01(rng);
            const glm::vec3 n(0, 0, 1);
            const float ct = u01(rng) < 0.1f ? 1e-4f * u01(rng) : u01(rng);
            const float ph = TWO_PI * u01(rng);
            const glm::vec3 wo(sqrtf(1 - ct * ct) * cosf(ph), sqrtf(1 - ct * ct) * sinf(ph), ct);
            glm::vec3 sn = glm::normalize(n + 0.6f * glm::vec3(u01(rng) - 0.5f, u01(rng) - 0.5f, 0));
            if (u01(rng) < 0.5f || glm::dot(sn, wo) <= 0) sn = n;
            const bool typical = ct > 0.1f && sn == n;
            c.s.normal = sn;
            c.s.coatNormal = n;
            PathSegment p{};
            p.ray.direction = -wo;
            p.color = glm::vec3(1);
            const bool outside = u01(rng) < 0.5f;

            {
                const float z = 1 - 2 * u01(rng);
                const float r = sqrtf(fmaxf(0, 1 - z * z));
                const float phi = TWO_PI * u01(rng);
                const BsdfEval e = evalPbr(wo, glm::vec3(r * cosf(phi), r * sinf(phi), z), n, outside, c.m, c.s);
                if (!(std::isfinite(e.fCos.x) && std::isfinite(e.fCos.y) && std::isfinite(e.fCos.z) && std::isfinite(e.pdf))
                    || e.fCos.x < 0 || e.fCos.y < 0 || e.fCos.z < 0 || e.pdf < 0)
                {
                    if (bad < 5) printf("bad eval %d: f cos %g %g %g pdf %g\n", i, e.fCos.x, e.fCos.y, e.fCos.z, e.pdf);
                    ++bad;
                }
            }

            if (scatterPbr(p, glm::vec3(0), n, outside, c.m, c.s, rng))
            {
                const glm::vec3 w = p.color;
                const glm::vec3 d = p.ray.direction;
                if (!(std::isfinite(w.x) && std::isfinite(w.y) && std::isfinite(w.z)) || w.x < 0 || w.y < 0 || w.z < 0
                    || !(std::isfinite(d.x) && std::isfinite(d.y) && std::isfinite(d.z)) || fabsf(glm::length(d) - 1) > 1e-3f
                    || !std::isfinite(p.pdf) || p.pdf < 0)
                {
                    if (bad < 5) printf("bad sample %d: w %g %g %g d %g %g %g pdf %g\n", i, w.x, w.y, w.z, d.x, d.y, d.z, p.pdf);
                    ++bad;
                }
                maxWeight = std::max(maxWeight, (double)maxComponent(w));
                const float mw = maxComponent(w);
                big += mw > 4.0f;
                if (typical) { bigTypical += mw > 4.0f; maxTypical = std::max(maxTypical, (double)mw); }
            }
        }
        const bool ok = check(bad == 0);
        printf("fuzz: %d bad samples, %d weights above 4, max weight %.3f; typical views: %d above 4, max %.3f%s\n", bad,
            big, maxWeight, bigTypical, maxTypical, ok ? "" : "  FAIL");
    }

    if (failures > 0)
    {
        printf("FAIL: %d of %d checks\n", failures, checks);
        return 1;
    }
    printf("PASS: %d checks\n", checks);
    return 0;
}

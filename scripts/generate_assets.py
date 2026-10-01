#!/usr/bin/env python3
"""Generate this project's original assets and profile its procedural meshes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'scenes' / 'assets'


def bmp(path, width, height, pixel):
    stride = (3 * width + 3) & ~3
    body = bytearray()
    for y in reversed(range(height)):
        for x in range(width):
            rgb = pixel(x / width, y / height)
            body.extend(bytes(reversed([max(0, min(255, round(c))) for c in rgb])))
        body.extend(b'\0' * (stride - 3 * width))
    header = b'BM' + struct.pack('<IHHI', 54 + len(body), 0, 0, 54)
    header += struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, len(body), 2835, 2835, 0, 0)
    path.write_bytes(header + body)


def _finite_vector(value):
    return all(math.isfinite(component) for component in value)


def _subtract(a, b):
    return tuple(left - right for left, right in zip(a, b))


def _dot(a, b):
    return sum(left * right for left, right in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _normalize(value, description):
    squared_length = _dot(value, value)
    if not _finite_vector(value) or not math.isfinite(squared_length) or squared_length <= 1e-24:
        raise ValueError(f'{description} is non-finite or degenerate')
    inverse_length = 1 / math.sqrt(squared_length)
    result = tuple(component * inverse_length for component in value)
    if not _finite_vector(result):
        raise ValueError(f'{description} did not produce a finite unit vector')
    return result


def centered_normal_surface(position, description):
    """Attach centered-partial normals to an analytic position function."""
    h = 1e-5

    def surface(u, v):
        p = position(u, v)
        pu = tuple((right - left) / (2 * h)
                   for right, left in zip(position(u + h, v), position(u - h, v)))
        pv = tuple((right - left) / (2 * h)
                   for right, left in zip(position(u, v + h), position(u, v - h)))
        if not _finite_vector(p) or not _finite_vector(pu) or not _finite_vector(pv):
            raise ValueError(f'{description} has a non-finite position or partial derivative')
        return p, _normalize(_cross(pv, pu), f'{description} normal')

    return surface


def validate_surface_obj(path, nu, nv, expected_triangles, expected_euler,
                         canonical_u, canonical_v, radial_cap_uv):
    """Validate the serialized OBJ, including exact welded topology."""
    position_values = []
    position_tokens = []
    uvs = []
    normal_values = []
    normal_tokens = []
    faces = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0] == '#':
            continue
        if fields[0] == 'v':
            if len(fields) != 4:
                raise ValueError(f'{path}:{line_number}: malformed position')
            position_tokens.append(tuple(fields[1:]))
            position_values.append(tuple(float(value) for value in fields[1:]))
        elif fields[0] == 'vt':
            if len(fields) != 3:
                raise ValueError(f'{path}:{line_number}: malformed UV')
            uvs.append(tuple(float(value) for value in fields[1:]))
        elif fields[0] == 'vn':
            if len(fields) != 4:
                raise ValueError(f'{path}:{line_number}: malformed normal')
            normal_tokens.append(tuple(fields[1:]))
            normal_values.append(tuple(float(value) for value in fields[1:]))
        elif fields[0] == 'f':
            if len(fields) != 4:
                raise ValueError(f'{path}:{line_number}: expected triangular faces')
            face = []
            for field in fields[1:]:
                indices = field.split('/')
                if len(indices) != 3:
                    raise ValueError(f'{path}:{line_number}: incomplete face index')
                face.append(tuple(int(index) - 1 for index in indices))
            faces.append(tuple(face))

    if len(faces) != expected_triangles:
        raise ValueError(f'{path}: expected {expected_triangles} triangles, found {len(faces)}')
    if not (len(position_values) == len(uvs) == len(normal_values)):
        raise ValueError(f'{path}: position, UV, and normal streams differ in length')
    finite_position_count = sum(_finite_vector(position) for position in position_values)
    finite_uv_count = sum(_finite_vector(uv) for uv in uvs)
    finite_normal_count = sum(_finite_vector(normal) for normal in normal_values)
    positions_finite = finite_position_count == len(position_values)
    uvs_finite = finite_uv_count == len(uvs)
    normals_finite = finite_normal_count == len(normal_values)
    if not positions_finite:
        raise ValueError(f'{path}: non-finite serialized position')
    if not uvs_finite:
        raise ValueError(f'{path}: non-finite serialized UV')
    uv_components = [component for uv in uvs for component in uv]
    uv_minimum = min(uv_components)
    uv_maximum = max(uv_components)
    uv_in_range_count = sum(all(0 <= component <= 1 for component in uv) for uv in uvs)
    if uv_minimum < 0 or uv_maximum > 1:
        raise ValueError(f'{path}: out-of-range serialized UV')
    if not normals_finite:
        raise ValueError(f'{path}: non-finite serialized normal')
    normal_lengths = [math.sqrt(_dot(normal, normal)) for normal in normal_values]
    maximum_normal_error = max(abs(length - 1) for length in normal_lengths)
    unit_normal_count = sum(abs(length - 1) <= 2e-7 for length in normal_lengths)
    if maximum_normal_error > 2e-7:
        raise ValueError(f'{path}: serialized normal is not unit length')

    grid_stride = nu + 1
    grid_count = grid_stride * (nv + 1)
    canonical_u_pair_count = 0
    if canonical_u:
        for j in range(nv + 1):
            first = j * grid_stride
            last = first + nu
            if (position_tokens[first] != position_tokens[last] or
                    normal_tokens[first] != normal_tokens[last]):
                raise ValueError(f'{path}: U seam is not exactly canonical')
            if uvs[first][0] != 0 or uvs[last][0] != 1 or uvs[first][1] != uvs[last][1]:
                raise ValueError(f'{path}: U seam did not retain UV endpoints')
            canonical_u_pair_count += 1
    canonical_v_pair_count = 0
    if canonical_v:
        for i in range(nu + 1):
            first = i
            last = nv * grid_stride + i
            if (position_tokens[first] != position_tokens[last] or
                    normal_tokens[first] != normal_tokens[last]):
                raise ValueError(f'{path}: V seam is not exactly canonical')
            if uvs[first][1] != 0 or uvs[last][1] != 1 or uvs[first][0] != uvs[last][0]:
                raise ValueError(f'{path}: V seam did not retain UV endpoints')
            canonical_v_pair_count += 1

    cap_ring_pair_count = 0
    if radial_cap_uv:
        for cap, j in enumerate((0, nv)):
            start = grid_count + cap * (nu + 2)
            if uvs[start] != (0.5, 0.5):
                raise ValueError(f'{path}: cap center UV is not radial-map center')
            expected_normal = (0.0, -1.0, 0.0) if j == 0 else (0.0, 1.0, 0.0)
            for i in range(nu + 1):
                cap_index = start + i + 1
                side_index = j * grid_stride + i
                if position_tokens[cap_index] != position_tokens[side_index]:
                    raise ValueError(f'{path}: cap and side ring positions are not exact')
                if normal_values[cap_index] != expected_normal:
                    raise ValueError(f'{path}: cap normal is not flat')
                theta = 0 if i == nu else 2 * math.pi * i / nu
                expected_uv = (0.5 + 0.5 * math.cos(theta),
                               0.5 + 0.5 * math.sin(theta))
                if any(abs(actual - expected) > 2e-15
                       for actual, expected in zip(uvs[cap_index], expected_uv)):
                    raise ValueError(f'{path}: cap ring UV is not radial')
                cap_ring_pair_count += 1

    welded_ids = {}
    edge_incidence = {}
    minimum_area = math.inf
    minimum_winding = math.inf
    positive_area_count = 0
    outward_winding_count = 0
    noncollapsed_triangle_count = 0
    for face in faces:
        welded_face = []
        face_positions = []
        face_normals = []
        for position_index, uv_index, normal_index in face:
            if not (0 <= position_index < len(position_values) and
                    0 <= uv_index < len(uvs) and
                    0 <= normal_index < len(normal_values)):
                raise ValueError(f'{path}: face index is out of range')
            token = position_tokens[position_index]
            welded_face.append(welded_ids.setdefault(token, len(welded_ids)))
            face_positions.append(position_values[position_index])
            face_normals.append(normal_values[normal_index])
        edge_a = _subtract(face_positions[1], face_positions[0])
        edge_b = _subtract(face_positions[2], face_positions[0])
        geometric = _cross(edge_a, edge_b)
        squared_twice_area = _dot(geometric, geometric)
        if not _finite_vector(geometric) or squared_twice_area <= 1e-24:
            raise ValueError(f'{path}: triangle has non-positive area')
        minimum_area = min(minimum_area, 0.5 * math.sqrt(squared_twice_area))
        positive_area_count += 1
        summed_normal = tuple(sum(normal[axis] for normal in face_normals)
                              for axis in range(3))
        winding = _dot(geometric, summed_normal)
        if not winding > 0:
            raise ValueError(f'{path}: triangle winding is not outward')
        minimum_winding = min(minimum_winding, winding)
        outward_winding_count += 1
        if len(set(welded_face)) != 3:
            raise ValueError(f'{path}: triangle collapses after exact welding')
        noncollapsed_triangle_count += 1
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((welded_face[first], welded_face[second])))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1

    bad_edges = [edge for edge, incidence in edge_incidence.items() if incidence != 2]
    if bad_edges:
        raise ValueError(f'{path}: welded mesh has {len(bad_edges)} non-manifold edges')
    euler = len(welded_ids) - len(edge_incidence) + len(faces)
    if euler != expected_euler:
        raise ValueError(f'{path}: expected Euler characteristic {expected_euler}, found {euler}')
    incidences = list(edge_incidence.values())
    return {
        'all_checks_passed': True,
        'counts': {
            'positions': len(position_values),
            'uvs': len(uvs),
            'normals': len(normal_values),
            'triangles': len(faces),
        },
        'finite': {
            'positions': positions_finite,
            'uvs': uvs_finite,
            'normals': normals_finite,
            'position_count': finite_position_count,
            'uv_count': finite_uv_count,
            'normal_count': finite_normal_count,
        },
        'normals': {
            'unit_length': True,
            'unit_length_count': unit_normal_count,
            'tolerance': 2e-7,
            'length_min': min(normal_lengths),
            'length_max': max(normal_lengths),
            'maximum_abs_unit_error': maximum_normal_error,
        },
        'uvs': {
            'within_unit_square': True,
            'within_unit_square_count': uv_in_range_count,
            'component_min': uv_minimum,
            'component_max': uv_maximum,
        },
        'triangles': {
            'positive_area': True,
            'positive_area_count': positive_area_count,
            'minimum_area': minimum_area,
            'outward_winding': True,
            'outward_winding_count': outward_winding_count,
            'minimum_winding_dot': minimum_winding,
            'noncollapsed_count': noncollapsed_triangle_count,
        },
        'seams': {
            'canonical_u_required': canonical_u,
            'canonical_u_position_normal_exact': True if canonical_u else None,
            'canonical_u_uv_endpoints_retained': True if canonical_u else None,
            'canonical_u_pair_count': canonical_u_pair_count,
            'canonical_u_expected_pairs': nv + 1 if canonical_u else 0,
            'canonical_v_required': canonical_v,
            'canonical_v_position_normal_exact': True if canonical_v else None,
            'canonical_v_uv_endpoints_retained': True if canonical_v else None,
            'canonical_v_pair_count': canonical_v_pair_count,
            'canonical_v_expected_pairs': nu + 1 if canonical_v else 0,
            'radial_caps_required': radial_cap_uv,
            'cap_side_positions_exact': True if radial_cap_uv else None,
            'cap_uvs_radial': True if radial_cap_uv else None,
            'cap_normals_flat': True if radial_cap_uv else None,
            'cap_ring_pair_count': cap_ring_pair_count,
            'cap_ring_expected_pairs': 2 * (nu + 1) if radial_cap_uv else 0,
        },
        'topology': {
            'welded_vertices': len(welded_ids),
            'welded_edges': len(edge_incidence),
            'faces': len(faces),
            'edge_incidence_min': min(incidences),
            'edge_incidence_max': max(incidences),
            'edge_incidence_two_count': sum(incidence == 2 for incidence in incidences),
            'bad_edge_count': len(bad_edges),
            'closed_two_manifold': not bad_edges,
            'euler_characteristic': euler,
            'expected_euler_characteristic': expected_euler,
        },
    }


def write_surface(path, nu, nv, surface, closed_v=False, *, canonical_u=False,
                  canonical_v=False, radial_cap_uv=False, expected_triangles=None,
                  expected_euler=None):
    if canonical_v and not closed_v:
        raise ValueError('canonical_v requires a V-closed surface')
    if radial_cap_uv and closed_v:
        raise ValueError('radial caps require an open V direction')
    new_mesh = canonical_u or canonical_v or radial_cap_uv or expected_triangles is not None
    format_value = (lambda value: f'{value:.17g}') if new_mesh else (lambda value: f'{value:.9g}')
    construction_start = time.perf_counter()
    vertices = []
    for j in range(nv + 1):
        for i in range(nu + 1):
            u = 0 if canonical_u and i == nu else i / nu
            v = 0 if canonical_v and j == nv else j / nv
            vertices.append(surface(u, v))
    lines = ['# Original parametric mesh generated by scripts/generate_assets.py']
    for p, n in vertices:
        lines.append('v ' + ' '.join(format_value(a) for a in p))
    for j in range(nv + 1):
        for i in range(nu + 1):
            lines.append('vt ' + format_value(i / nu) + ' ' + format_value(j / nv))
    for p, n in vertices:
        lines.append('vn ' + ' '.join(format_value(a) for a in n))
    triangle_count = 0
    def face(a, b, c):
        nonlocal triangle_count
        lines.append('f ' + ' '.join(f'{k}/{k}/{k}' for k in (a, b, c)))
        triangle_count += 1
    for j in range(nv):
        for i in range(nu):
            a = j * (nu + 1) + i + 1
            b, d = a + 1, a + nu + 1
            face(a, d, b)
            face(b, d, d + 1)
    if not closed_v:
        # Cap with separate normals and UVs; sides retain their smooth normals.
        for j, normal in ((0, (0, -1, 0)), (nv, (0, 1, 0))):
            center = (0, vertices[j * (nu + 1)][0][1], 0)
            start = len(vertices) + 1
            cap_positions = [center] + [vertices[j * (nu + 1) + i][0]
                                        for i in range(nu + 1)]
            for cap_index, p in enumerate(cap_positions):
                vertices.append((p, normal))
                if new_mesh:
                    position = 'v ' + ' '.join(format_value(value) for value in p)
                    if radial_cap_uv and cap_index > 0:
                        ring_index = cap_index - 1
                        theta = 0 if ring_index == nu else 2 * math.pi * ring_index / nu
                        uv = (0.5 + 0.5 * math.cos(theta),
                              0.5 + 0.5 * math.sin(theta))
                    else:
                        uv = (0.5, 0.5)
                    texture = 'vt ' + ' '.join(format_value(value) for value in uv)
                    normal_line = 'vn ' + ' '.join(format_value(value) for value in normal)
                    lines.extend((position, texture, normal_line))
                else:
                    lines.extend(('v ' + ' '.join(map(str, p)), 'vt 0.5 0.5', 'vn ' + ' '.join(map(str, normal))))
            for i in range(nu):
                if j == 0:
                    face(start, start + i + 1, start + i + 2)
                else:
                    face(start, start + i + 2, start + i + 1)
    path.write_text('\n'.join(lines) + '\n')
    construction_write_ms = (time.perf_counter() - construction_start) * 1000
    byte_size = path.stat().st_size
    validation = None
    validation_ms = None
    if expected_triangles is not None or expected_euler is not None:
        if expected_triangles is None or expected_euler is None:
            raise ValueError('mesh validation requires triangle and Euler expectations')
        validation_start = time.perf_counter()
        validation = validate_surface_obj(path, nu, nv, expected_triangles, expected_euler,
                                          canonical_u, canonical_v, radial_cap_uv)
        validation_ms = (time.perf_counter() - validation_start) * 1000
    return {
        'construction_write_ms': construction_write_ms,
        'validation_ms': validation_ms,
        'byte_size': byte_size,
        'triangle_count': triangle_count,
        'validation': validation,
    }


def vase(u, v):
    theta = 2 * math.pi * u
    r = .43 + .13 * math.sin(math.pi * v) + .17 * math.sin(3 * math.pi * v) ** 2
    dr = .13 * math.pi * math.cos(math.pi * v) + .51 * math.pi * math.sin(6 * math.pi * v)
    n = (math.cos(theta), -dr / 2.7, math.sin(theta))
    length = math.sqrt(sum(a * a for a in n))
    return (r * math.cos(theta), 2.7 * v, r * math.sin(theta)), tuple(a / length for a in n)


def torus(u, v):
    a, b = 2 * math.pi * u, 2 * math.pi * v
    r, tube = .82, .25
    p = ((r + tube * math.cos(b)) * math.cos(a), tube * math.sin(b), (r + tube * math.cos(b)) * math.sin(a))
    n = (math.cos(b) * math.cos(a), math.sin(b), math.cos(b) * math.sin(a))
    return p, n


def fluted_body_position(u, v):
    theta = 2 * math.pi * u
    radius_profile = (.50 + .14 * math.sin(math.pi * v) +
                      .09 * math.sin(3 * math.pi * v) ** 2)
    radius = radius_profile * (1 + .18 * math.cos(7 * theta))
    axis_x = 1 + .16 * math.sin(2 * math.pi * v)
    axis_z = 1 - .12 * math.sin(2 * math.pi * v)
    phi = theta + 1.5 * math.pi * v + .20 * math.sin(2 * math.pi * v)
    return (axis_x * radius * math.cos(phi),
            2.7 * v,
            axis_z * radius * math.sin(phi))


def swept_knot_position(u, v):
    t = 2 * math.pi * u
    theta = 2 * math.pi * v
    amplitude = 1 + .35 * math.cos(3 * t)
    amplitude_first = -1.05 * math.sin(3 * t)
    amplitude_second = -3.15 * math.cos(3 * t)
    cos_two_t = math.cos(2 * t)
    sin_two_t = math.sin(2 * t)
    center = (amplitude * cos_two_t,
              amplitude * sin_two_t,
              .35 * math.sin(3 * t))
    center_first = (amplitude_first * cos_two_t - 2 * amplitude * sin_two_t,
                    amplitude_first * sin_two_t + 2 * amplitude * cos_two_t,
                    1.05 * math.cos(3 * t))
    center_second = (amplitude_second * cos_two_t -
                     4 * amplitude_first * sin_two_t - 4 * amplitude * cos_two_t,
                     amplitude_second * sin_two_t +
                     4 * amplitude_first * cos_two_t - 4 * amplitude * sin_two_t,
                     -3.15 * math.sin(3 * t))
    tangent = _normalize(center_first, 'swept-knot tangent')
    normal_component = _dot(center_second, tangent)
    normal = _normalize(tuple(second - normal_component * tangent_component
                              for second, tangent_component in zip(center_second, tangent)),
                        'swept-knot principal normal')
    binormal = _cross(tangent, normal)
    radius = .10 + .025 * math.sin(5 * t)
    radial = tuple(math.cos(theta) * normal_component +
                   math.sin(theta) * binormal_component
                   for normal_component, binormal_component in zip(normal, binormal))
    return tuple(center_component + radius * radial_component
                 for center_component, radial_component in zip(center, radial))


def new_mesh_jobs():
    fluted_body = centered_normal_surface(fluted_body_position, 'fluted-body surface')
    swept_knot = centered_normal_surface(swept_knot_position, 'swept-knot surface')
    jobs = []
    for nu, nv, lod in ((32, 24, 'low'), (64, 48, 'medium'), (128, 96, 'high')):
        jobs.append({
            'shape': 'fluted-body',
            'lod': lod,
            'nu': nu,
            'nv': nv,
            'path': ASSETS / f'fluted-body-{lod}.obj',
            'surface': fluted_body,
            'closed_v': False,
            'canonical_u': True,
            'canonical_v': False,
            'radial_cap_uv': True,
            'expected_triangles': 2 * nu * nv + 2 * nu,
            'expected_euler': 2,
        })
    for nu, nv, lod in ((64, 8, 'low'), (128, 16, 'medium'), (256, 32, 'high')):
        jobs.append({
            'shape': 'swept-knot',
            'lod': lod,
            'nu': nu,
            'nv': nv,
            'path': ASSETS / f'swept-knot-{lod}.obj',
            'surface': swept_knot,
            'closed_v': True,
            'canonical_u': True,
            'canonical_v': True,
            'radial_cap_uv': False,
            'expected_triangles': 2 * nu * nv,
            'expected_euler': 0,
        })
    return jobs


def generate_new_mesh(job, output_path=None):
    path = job['path'] if output_path is None else Path(output_path)
    return write_surface(
        path, job['nu'], job['nv'], job['surface'], job['closed_v'],
        canonical_u=job['canonical_u'], canonical_v=job['canonical_v'],
        radial_cap_uv=job['radial_cap_uv'],
        expected_triangles=job['expected_triangles'], expected_euler=job['expected_euler'])


def timing_summary(samples):
    if not samples or any(not math.isfinite(sample) or sample < 0 for sample in samples):
        raise ValueError('profiling produced invalid wall-clock samples')
    return {
        'samples_ms': [round(sample, 6) for sample in samples],
        'minimum_ms': round(min(samples), 6),
        'median_ms': round(statistics.median(samples), 6),
        'mean_ms': round(statistics.fmean(samples), 6),
        'maximum_ms': round(max(samples), 6),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_profile_document(document):
    if document.get('schema_version') != 2:
        raise ValueError('profile schema_version must be 2')
    if (document.get('profile_type') != 'cpu_mesh_generation' or
            document.get('purpose') != 'cpu-procedural-generation-profile' or
            document.get('warmups') != 1 or document.get('measured_repeats') != 3 or
            document.get('units') != 'milliseconds'):
        raise ValueError('profile run metadata is inconsistent')
    records = document.get('records')
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError('profile must contain exactly six mesh records')
    identities = set()
    expected_paths = {
        ('fluted-body', 'low'): 'scenes/assets/fluted-body-low.obj',
        ('fluted-body', 'medium'): 'scenes/assets/fluted-body-medium.obj',
        ('fluted-body', 'high'): 'scenes/assets/fluted-body-high.obj',
        ('swept-knot', 'low'): 'scenes/assets/swept-knot-low.obj',
        ('swept-knot', 'medium'): 'scenes/assets/swept-knot-medium.obj',
        ('swept-knot', 'high'): 'scenes/assets/swept-knot-high.obj',
    }
    for mesh in records:
        identity = (mesh.get('shape'), mesh.get('lod'))
        if identity in identities:
            raise ValueError(f'profile contains duplicate mesh record {identity}')
        identities.add(identity)
        if mesh.get('relative_path') != expected_paths.get(identity):
            raise ValueError(f'profile mesh {identity} has an invalid canonical relative path')
        if mesh.get('nu', 0) <= 0 or mesh.get('nv', 0) <= 0:
            raise ValueError(f'profile mesh {identity} has invalid tessellation dimensions')
        if mesh.get('triangles', 0) <= 0:
            raise ValueError(f'profile mesh {identity} has invalid output counts')
        for size_field in ('measured_generated_byte_size', 'canonical_asset_byte_size'):
            if not isinstance(mesh.get(size_field), int) or mesh[size_field] <= 0:
                raise ValueError(f'profile mesh {identity} has invalid {size_field}')
        for digest_field in ('measured_generated_sha256', 'canonical_asset_sha256'):
            digest = mesh.get(digest_field, '')
            if len(digest) != 64:
                raise ValueError(f'profile mesh {identity} has invalid {digest_field}')
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(f'profile mesh {identity} has invalid {digest_field}') from error
        byte_identical = mesh.get('byte_identical_to_canonical')
        if not isinstance(byte_identical, bool):
            raise ValueError(f'profile mesh {identity} lacks a byte-comparison result')
        if (byte_identical and
                (mesh['measured_generated_byte_size'] != mesh['canonical_asset_byte_size'] or
                 mesh['measured_generated_sha256'] != mesh['canonical_asset_sha256'])):
            raise ValueError(f'profile mesh {identity} has inconsistent byte-comparison metadata')
        runs = mesh.get('measured_runs')
        if not isinstance(runs, list) or len(runs) != 3:
            raise ValueError(f'profile mesh {identity} must contain three measured runs')
        for run in [mesh.get('warmup_run'), *runs]:
            if not isinstance(run, dict):
                raise ValueError(f'profile mesh {identity} has a malformed run record')
            for field in ('construction_write_ms', 'validation_ms'):
                value = run.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'profile mesh {identity} has invalid {field}')
            if run.get('generated_byte_size') != mesh['measured_generated_byte_size']:
                raise ValueError(f'profile mesh {identity} run byte sizes disagree')
            if run.get('generated_sha256') != mesh['measured_generated_sha256']:
                raise ValueError(f'profile mesh {identity} run digests disagree')
        measured_generation = [run['construction_write_ms'] for run in runs]
        if mesh.get('generation_and_write_ms') != measured_generation:
            raise ValueError(f'profile mesh {identity} timing samples disagree')
        validation = mesh.get('validation')
        if not isinstance(validation, dict) or not validation.get('all_checks_passed'):
            raise ValueError(f'profile mesh {identity} did not pass validation')
        if validation['counts']['triangles'] != mesh['triangles']:
            raise ValueError(f'profile mesh {identity} triangle counts disagree')
        counts = validation['counts']
        finite = validation['finite']
        normals = validation['normals']
        uvs = validation['uvs']
        triangles = validation['triangles']
        seams = validation['seams']
        if (finite['position_count'] != counts['positions'] or
                finite['uv_count'] != counts['uvs'] or
                finite['normal_count'] != counts['normals'] or
                normals['unit_length_count'] != counts['normals'] or
                uvs['within_unit_square_count'] != counts['uvs'] or
                triangles['positive_area_count'] != counts['triangles'] or
                triangles['outward_winding_count'] != counts['triangles'] or
                triangles['noncollapsed_count'] != counts['triangles']):
            raise ValueError(f'profile mesh {identity} validation counts are incomplete')
        if (seams['canonical_u_pair_count'] != seams['canonical_u_expected_pairs'] or
                seams['canonical_v_pair_count'] != seams['canonical_v_expected_pairs'] or
                seams['cap_ring_pair_count'] != seams['cap_ring_expected_pairs']):
            raise ValueError(f'profile mesh {identity} seam counts disagree')
        topology = validation['topology']
        if (not topology['closed_two_manifold'] or topology['edge_incidence_min'] != 2 or
                topology['edge_incidence_max'] != 2 or topology['bad_edge_count'] != 0 or
                topology['edge_incidence_two_count'] != topology['welded_edges'] or
                topology['faces'] != counts['triangles'] or
                topology['euler_characteristic'] != topology['expected_euler_characteristic']):
            raise ValueError(f'profile mesh {identity} is not a closed two-manifold')
    if identities != set(expected_paths):
        raise ValueError('profile mesh identities do not match the six required tessellations')


def profile_new_meshes(output_json, profile_only, canonical_asset_directory=None):
    jobs = new_mesh_jobs()
    output_path = Path(output_json)
    resolved_output = output_path.resolve()
    if any(resolved_output == job['path'].resolve() for job in jobs):
        raise ValueError('profile JSON path must not overwrite a generated OBJ')
    try:
        resolved_output.relative_to((ROOT / 'scenes').resolve())
    except ValueError:
        pass
    else:
        raise ValueError('profile JSON path must not overwrite or enter the scenes tree')

    comparison_directory = (ASSETS if canonical_asset_directory is None
                            else Path(canonical_asset_directory))
    mesh_profiles = []
    with tempfile.TemporaryDirectory(prefix='cis5650-procedural-profile-') as temporary:
        temporary_directory = Path(temporary)
        for job in jobs:
            canonical_path = comparison_directory / job['path'].name
            if not canonical_path.is_file():
                raise FileNotFoundError(
                    f'canonical comparison asset does not exist: {canonical_path}')
            canonical_bytes = canonical_path.read_bytes()
            canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
            scratch_path = temporary_directory / job['path'].name

            def measured_generation():
                result = generate_new_mesh(job, scratch_path)
                result['generated_sha256'] = file_sha256(scratch_path)
                return result

            warmup = measured_generation()
            measured = [measured_generation() for _ in range(3)]
            all_runs = [warmup, *measured]
            if any(run['triangle_count'] != job['expected_triangles'] for run in all_runs):
                raise ValueError(f"{job['path']}: generated triangle count changed between runs")
            generated_outputs = {(run['byte_size'], run['generated_sha256'])
                                 for run in all_runs}
            if len(generated_outputs) != 1:
                raise ValueError(f"{job['path']}: generated bytes changed between runs")
            reference_validation = warmup['validation']
            if any(run['validation'] != reference_validation for run in measured):
                raise ValueError(f"{job['path']}: validation summary changed between runs")

            measured_bytes = scratch_path.read_bytes()
            measured_sha256 = hashlib.sha256(measured_bytes).hexdigest()
            if (len(measured_bytes) != measured[-1]['byte_size'] or
                    measured_sha256 != measured[-1]['generated_sha256']):
                raise ValueError(f"{job['path']}: final temporary output metadata changed")

            def run_record(run_index, values):
                return {
                    'run': run_index,
                    'construction_write_ms': round(values['construction_write_ms'], 6),
                    'validation_ms': round(values['validation_ms'], 6),
                    'generated_byte_size': values['byte_size'],
                    'generated_sha256': values['generated_sha256'],
                }

            construction_samples = [run['construction_write_ms'] for run in measured]
            validation_samples = [run['validation_ms'] for run in measured]
            construction_timing = timing_summary(construction_samples)
            validation_timing = timing_summary(validation_samples)
            mesh_profiles.append({
                'shape': job['shape'],
                'lod': job['lod'],
                'nu': job['nu'],
                'nv': job['nv'],
                'relative_path': job['path'].relative_to(ROOT).as_posix(),
                'triangles': measured[-1]['validation']['counts']['triangles'],
                'measured_generated_byte_size': len(measured_bytes),
                'measured_generated_sha256': measured_sha256,
                'canonical_asset_byte_size': len(canonical_bytes),
                'canonical_asset_sha256': canonical_sha256,
                'byte_identical_to_canonical': measured_bytes == canonical_bytes,
                'warmup_run': run_record(0, warmup),
                'measured_runs': [run_record(index, values)
                                  for index, values in enumerate(measured, 1)],
                'generation_and_write_ms': construction_timing['samples_ms'],
                'generation_and_write_median_ms': construction_timing['median_ms'],
                'validation_ms': validation_timing['median_ms'],
                'construction_write_timing': construction_timing,
                'post_write_validation_timing': validation_timing,
                'validation': measured[-1]['validation'],
            })

    clock = time.get_clock_info('perf_counter')
    document = {
        'schema_version': 2,
        'profile_type': 'cpu_mesh_generation',
        'purpose': 'cpu-procedural-generation-profile',
        'profile_only': profile_only,
        'warmups': 1,
        'measured_repeats': 3,
        'units': 'milliseconds',
        'note': ('Measured CPU wall-clock procedural-generation profile for this invocation; '
                 'interpret results with the recorded host metadata and caveats.'),
        'generator': {
            'implementation': 'Python standard-library CPU procedural mesh generator',
            'gpu_generation': False,
            'timed_scope': ('surface construction, OBJ serialization, and write_text wall time; '
                            'temporary-directory setup, file stat, hashing, canonical reads, '
                            'comparison, and post-write validation are excluded'),
            'validation_scope': 'serialized OBJ validation, timed and reported separately',
            'output_scope': 'warmup and measured OBJ files exist only in a TemporaryDirectory',
            'canonical_asset_scope': 'read-only byte-size, SHA-256, and exact-byte comparison',
            'warmup_runs': 1,
            'measured_runs': 3,
        },
        'machine': {
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'python_compiler': platform.python_compiler(),
            'system': platform.system(),
            'release': platform.release(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor() or 'unreported',
        },
        'clock': {
            'name': 'time.perf_counter',
            'implementation': clock.implementation,
            'resolution_seconds': clock.resolution,
            'monotonic': clock.monotonic,
            'adjustable': clock.adjustable,
        },
        'caveats': [
            ('Wall-clock results describe the recorded hardware, Python runtime, operating '
             'system, filesystem, host load, cache state, and this process invocation.'),
            ('Each sample measures CPU construction, OBJ serialization, and one OS-buffered '
             'temporary-file write; Path.write_text does not include an fsync.'),
            ('Validation, hashing, canonical-asset reads and comparisons, and report emission '
             'are outside the generation-and-write interval.'),
            ('A false byte_identical_to_canonical value can reflect platform math-library '
             'formatting differences and does not invalidate an otherwise valid mesh.'),
            'No GPU geometry generation is performed or measured.',
        ],
        'records': mesh_profiles,
    }
    validate_profile_document(document)
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized)
    loaded = json.loads(output_path.read_text())
    validate_profile_document(loaded)
    if loaded != document:
        raise ValueError('profile JSON did not round-trip exactly')
    return document


def obj(kind, material, pos, scale=(1, 1, 1), rot=(0, 0, 0), file=None):
    o = dict(TYPE=kind, MATERIAL=material, TRANS=list(pos), SCALE=list(scale), ROTAT=list(rot))
    if file:
        o['FILE'] = file
    return o


def scene(name, materials, objects, **camera_overrides):
    camera = dict(RES=[960, 640], FOVY=22, ITERATIONS=2048, DEPTH=10, FILE=name,
                  EYE=[4.8, 2.8, 7.2], LOOKAT=[0, 1.2, 0], UP=[0, 1, 0], APERTURE=0,
                  FOCUS=8.8)
    camera.update(camera_overrides)
    data = dict(Materials=materials, Objects=objects, Camera=camera, BVH_MAX_DEPTH=32)
    (ROOT / 'scenes' / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    return data


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for nu, nv, label in ((16, 8, 'low'), (32, 16, 'medium'), (64, 32, 'high')):
        write_surface(ASSETS / f'vase-{label}.obj', nu, nv, vase)
    write_surface(ASSETS / 'torus.obj', 64, 24, torus, True)
    for job in new_mesh_jobs():
        generate_new_mesh(job)
    def glaze(u, v):
        stripe = .5 + .5 * math.sin(2 * math.pi * (8 * u + 2 * v))
        band = .5 + .5 * math.cos(2 * math.pi * 7 * v)
        return (18 + 24 * stripe, 82 + 90 * band, 130 + 110 * stripe)
    def height(u, v):
        h = 128 + 65 * math.sin(2 * math.pi * 16 * u) * math.cos(2 * math.pi * 12 * v)
        return h, h, h
    bmp(ASSETS / 'glaze.bmp', 256, 256, glaze)
    bmp(ASSETS / 'height.bmp', 256, 256, height)
    (ASSETS / 'area-light.obj').write_text('# Original downward-facing triangle light\nv -1 0 -1\nv 1 0 -1\nv 0 0 1\nvt 0 0\nvt 1 0\nvt 0.5 1\nvn 0 -1 0\nf 1/1/1 2/2/1 3/3/1\n')
    mats = {
        'ceramic': dict(TYPE='Diffuse', RGB=[.95, .95, .95], TEXTURE='assets/glaze.bmp', BUMP='assets/height.bmp', BUMP_STRENGTH=.025),
        'glass': dict(TYPE='Glass', RGB=[.99, .995, 1], IOR=1.5),
        'gold': dict(TYPE='Specular', RGB=[.93, .68, .28], ROUGHNESS=0),
        'floor': dict(TYPE='Diffuse', RGB=[.7, .73, .8], PROCEDURAL='checker'),
        'wall': dict(TYPE='Diffuse', RGB=[.08, .115, .18]),
        'warm': dict(TYPE='Emitting', RGB=[1, .82, .62], EMITTANCE=16),
        'cool': dict(TYPE='Emitting', RGB=[.5, .72, 1], EMITTANCE=12),
    }
    objects = [obj('cube', 'floor', (0, -.2, 0), (18, .4, 18)),
               obj('cube', 'wall', (0, 3, -3.2), (14, 6, .2)),
               obj('obj', 'ceramic', (-2, 0, 0), file='assets/vase-high.obj'),
               obj('sphere', 'glass', (0, 1.1, .5), (2.2, 2.2, 2.2)),
               obj('obj', 'gold', (2.1, 1.2, -.1), rot=(65, 0, 18), file='assets/torus.obj'),
               obj('cube', 'warm', (-2, 5.3, 2), (4, .15, 2)),
               obj('cube', 'cool', (4, 3.8, -.5), (.15, 3, 2))]
    scene('optical-workbench', mats, objects)
    scene('preview-smoke', mats, objects, RES=[192, 128], ITERATIONS=16)
    scene('workbench-dof', mats, objects, APERTURE=.22, FOCUS=8.8)
    plain = json.loads(json.dumps(mats))
    plain['ceramic'].pop('BUMP')
    plain['ceramic'].pop('BUMP_STRENGTH')
    scene('workbench-no-bump', plain, objects)
    procedural = json.loads(json.dumps(plain))
    procedural['ceramic'].pop('TEXTURE')
    procedural['ceramic']['PROCEDURAL'] = 'checker'
    scene('workbench-procedural', procedural, objects)
    opaque = json.loads(json.dumps(mats))
    opaque['glass'] = dict(TYPE='Specular', RGB=[.99, .995, 1])
    scene('workbench-mirror', opaque, objects)
    for level in ('low', 'medium', 'high'):
        variants = json.loads(json.dumps(objects))
        variants[2]['FILE'] = f'assets/vase-{level}.obj'
        variants.pop(4)
        scene(f'complexity-{level}', mats, variants, RES=[96, 64], ITERATIONS=8, DEPTH=6)
    uniform = json.loads(json.dumps(mats))
    uniform['uniform'] = dict(TYPE='Diffuse', RGB=[.7, .73, .8])
    uniform_objects = json.loads(json.dumps(objects))
    for o in uniform_objects:
        if o['MATERIAL'] not in ('warm', 'cool'):
            o['MATERIAL'] = 'uniform'
    scene('uniform-benchmark', uniform, uniform_objects, RES=[192, 128], ITERATIONS=16, DEPTH=8)
    # A closed diffuse chamber creates long paths for compaction analysis.
    closed = [obj('cube', 'wall', (0, 3, -4), (12, 6, .1)),
              obj('cube', 'wall', (0, 3, 4), (12, 6, .1)),
              obj('cube', 'floor', (0, -.05, 0), (12, .1, 8)),
              obj('cube', 'floor', (0, 6, 0), (12, .1, 8)),
              obj('cube', 'wall', (-6, 3, 0), (.1, 6, 8)),
              obj('cube', 'wall', (6, 3, 0), (.1, 6, 8)),
              obj('cube', 'warm', (0, 5.9, 0), (2, .1, 2)),
              obj('sphere', 'ceramic', (0, 1, 0), (2, 2, 2))]
    scene('closed-benchmark', mats, closed, RES=[192, 128], ITERATIONS=16, DEPTH=12, EYE=[0, 3, 3.8], LOOKAT=[0, 2.8, 0])
    scene('open-benchmark', mats, [objects[i] for i in (0, 3, 5)], RES=[192, 128], ITERATIONS=16, DEPTH=12)
    # Isolate each light distribution, including nonuniform scale, for MIS tests.
    light_materials = {'surface': dict(TYPE='Diffuse', RGB=[.7, .7, .7]),
                       'light': dict(TYPE='Emitting', RGB=[1, 1, 1], EMITTANCE=8)}
    base_objects = [obj('cube', 'surface', (0, -.1, 0), (10, .2, 10)),
                    obj('cube', 'surface', (0, 2, -3), (10, 4, .1))]
    for label, emitter in [('cube', obj('cube', 'light', (0, 3.8, 0), (3, .1, 2), (0, 25, 0))),
                           ('sphere', obj('sphere', 'light', (0, 3.8, 0), (3, 1, 2), (0, 25, 0))),
                           ('mesh', obj('obj', 'light', (0, 3.8, 0), (1.7, 1, 1.2), (0, 25, 0), 'assets/area-light.obj'))]:
        scene('light-test-' + label, light_materials, base_objects + [emitter], RES=[64, 48], ITERATIONS=4096, DEPTH=4,
              EYE=[4, 2.5, 6], LOOKAT=[0, .5, 0], FOVY=22)

    def procedural_showcase_materials(procedural):
        return {
            'shape': dict(TYPE='Diffuse', RGB=[.95, .95, .95], PROCEDURAL=procedural),
            'floor': dict(TYPE='Diffuse', RGB=[.7, .73, .8], PROCEDURAL='checker'),
            'wall': dict(TYPE='Diffuse', RGB=[.08, .115, .18]),
            'warm': dict(TYPE='Emitting', RGB=[1, .82, .62], EMITTANCE=16),
            'cool': dict(TYPE='Emitting', RGB=[.5, .72, 1], EMITTANCE=12),
        }

    def procedural_showcase_objects(asset, position):
        return [obj('cube', 'floor', (0, -.2, 0), (18, .4, 18)),
                obj('cube', 'wall', (0, 3, -3.2), (14, 6, .2)),
                obj('obj', 'shape', position, file=asset),
                obj('cube', 'warm', (-2, 5.3, 2), (4, .15, 2)),
                obj('cube', 'cool', (4, 3.8, -.5), (.15, 3, 2))]

    for procedural_name in ('none', 'marble', 'cells'):
        scene(f'procedural-fluted-body-{"plain" if procedural_name == "none" else procedural_name}',
              procedural_showcase_materials(procedural_name),
              procedural_showcase_objects('assets/fluted-body-high.obj', (0, 0, 0)),
              RES=[640, 424], ITERATIONS=512, DEPTH=10,
              EYE=[2.3, 1.9, 4.2], LOOKAT=[0, 1.35, 0], FOVY=18)
        scene(f'procedural-swept-knot-{"plain" if procedural_name == "none" else procedural_name}',
              procedural_showcase_materials(procedural_name),
              procedural_showcase_objects('assets/swept-knot-high.obj', (0, 1.5, 0)),
              RES=[640, 424], ITERATIONS=512, DEPTH=10,
              EYE=[2.8, 2.3, 5.2], LOOKAT=[0, 1.5, 0], FOVY=18)

    still_life_materials = {
        'body': dict(TYPE='Diffuse', RGB=[.95, .95, .95], PROCEDURAL='marble'),
        'knot': dict(TYPE='Diffuse', RGB=[.95, .95, .95], PROCEDURAL='cells'),
        'glass': dict(TYPE='Glass', RGB=[.99, .995, 1], IOR=1.5),
        'floor': dict(TYPE='Diffuse', RGB=[.7, .73, .8], PROCEDURAL='checker'),
        'wall': dict(TYPE='Diffuse', RGB=[.08, .115, .18]),
        'warm': dict(TYPE='Emitting', RGB=[1, .82, .62], EMITTANCE=16),
        'cool': dict(TYPE='Emitting', RGB=[.5, .72, 1], EMITTANCE=12),
    }
    still_life_objects = [obj('cube', 'floor', (0, -.2, 0), (18, .4, 18)),
                          obj('cube', 'wall', (0, 3, -3.2), (14, 6, .2)),
                          obj('obj', 'body', (-2, 0, 0), file='assets/fluted-body-high.obj'),
                          obj('obj', 'knot', (2.3, 1.5, -.15), file='assets/swept-knot-high.obj'),
                          obj('sphere', 'glass', (0, .65, 1), (1.3, 1.3, 1.3)),
                          obj('cube', 'warm', (-2, 5.3, 2), (4, .15, 2)),
                          obj('cube', 'cool', (4, 3.8, -.5), (.15, 3, 2))]
    scene('procedural-optical-stilllife', still_life_materials, still_life_objects,
          RES=[960, 640], ITERATIONS=2048, DEPTH=10,
          EYE=[4.8, 2.8, 7.2], LOOKAT=[0, 1.2, 0], FOVY=22)
    print('Generated original and procedural OBJ/BMP assets and reproducible scene variants.')


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Generate deterministic Project 3 assets and optional CPU profiles.')
    parser.add_argument(
        '--profile', metavar='OUTPUT_JSON',
        help='write a stdlib-only CPU generation profile for the six procedural meshes')
    parser.add_argument(
        '--profile-only', action='store_true',
        help=('profile six temporary OBJ outputs without regenerating checked-in assets; '
              'requires --profile'))
    arguments = parser.parse_args(argv)
    if arguments.profile_only and arguments.profile is None:
        parser.error('--profile-only requires --profile OUTPUT_JSON')
    return arguments


def cli(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if not arguments.profile_only:
        main()
    if arguments.profile is not None:
        profile_new_meshes(arguments.profile, arguments.profile_only)
        print(f'Wrote CPU procedural-generation profile to {arguments.profile}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())

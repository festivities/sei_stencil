import bpy
import gpu
import os
import numpy as np

bl_info = {
    "name": "Sei Stencil",
    "author": "Seilotte",
    "version": (2, 0, 0),
    "blender": (5, 0, 0),
    "location": "3D View > Properties > Sei",
    "description": "Creates stencil passes for the viewport and final render.",
    "tracker_url": "https://github.com/seilotte/Blender-Stuff/tree/main/Add-ons/sei_stencil",
    "doc_url": "https://github.com/seilotte/Blender-Stuff/issues",
    "category": "Workflow",
}

IMAGE_NAME_PREFIX = '_SSTENCIL'
MAX_STENCIL_PASSES = 8
DEBUG_MODE = True

def get_image_name(pass_index):
    return f'{IMAGE_NAME_PREFIX}_{pass_index}'

# ===========================
# Property Groups
# ===========================

class SEI_PG_stencil_collection(bpy.types.PropertyGroup):
    collection: bpy.props.PointerProperty(
        type = bpy.types.Collection,
        name = 'Collection',
        description = 'Collection to retrieve objects from'
    )
    stencil_pass: bpy.props.IntProperty(
        name = 'Pass',
        description = 'Which stencil pass (0-7) this collection targets',
        min = 0, max = MAX_STENCIL_PASSES - 1, default = 0
    )

# ===========================
# UI List
# ===========================

class SEI_UL_stencil_collections(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, 'collection', text='')
            row.prop(item, 'stencil_pass', text='Pass')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text='', icon='OUTLINER_COLLECTION')

# ===========================
# List Management Operators
# ===========================

class SEI_OT_stencil_collection_add(bpy.types.Operator):
    bl_idname = 'sei.stencil_collection_add'
    bl_label = 'Add Stencil Collection'
    bl_description = 'Add a new collection entry to the stencil list'

    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.stencil_collections.add()
        context.scene.stencil_collections_active_index = \
            len(context.scene.stencil_collections) - 1
        return {'FINISHED'}

class SEI_OT_stencil_collection_remove(bpy.types.Operator):
    bl_idname = 'sei.stencil_collection_remove'
    bl_label = 'Remove Stencil Collection'
    bl_description = 'Remove the selected collection entry from the stencil list'

    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.stencil_collections) > 0

    def execute(self, context):
        idx = context.scene.stencil_collections_active_index
        context.scene.stencil_collections.remove(idx)
        context.scene.stencil_collections_active_index = max(0, idx - 1)
        return {'FINISHED'}

# ===========================
# Helper Functions
# ===========================

def get_pass_collections(scene):
    """Group collections by stencil pass index.
    Returns dict: pass_idx -> [collection, ...]"""
    pass_collections = {}
    for item in scene.stencil_collections:
        if item.collection is None:
            continue
        pass_idx = item.stencil_pass
        if pass_idx not in pass_collections:
            pass_collections[pass_idx] = []
        pass_collections[pass_idx].append(item.collection)
    return pass_collections

def has_any_stencil_objects(scene):
    """Check if any valid collection with objects is assigned."""
    for item in scene.stencil_collections:
        if item.collection and item.collection.all_objects:
            return True
    return False

def build_batches_for_collections(collections, depsgraph):
    """Build GPU batches for all mesh objects in the given collections."""
    batches_matrices = []

    for coll in collections:
        for obj in coll.all_objects:
            if obj.type != 'MESH' \
            or obj.visible_get() is False:
                continue

            # TODO: Use CBlenderMalt for better performance.
            mesh = obj.evaluated_get(depsgraph).data

            if mesh is None \
            or len(mesh.polygons) < 1:
                continue

            vcol = mesh.attributes.get(mesh.attributes.default_color_name)

            if vcol is None:
                vertices = np.empty((len(mesh.vertices), 3), 'f')
                indices = np.empty((len(mesh.loop_triangles), 3), 'i')
                colours = np.ones((len(mesh.vertices), 3), 'f') # white

                mesh.vertices.foreach_get('co', vertices.ravel())
                mesh.loop_triangles.foreach_get('vertices', indices.ravel())

            elif vcol.domain == 'POINT':
                vertices = np.empty((len(mesh.vertices), 3), 'f')
                indices = np.empty((len(mesh.loop_triangles), 3), 'i')
                colours = np.empty((len(vcol.data), 4), 'f')

                mesh.vertices.foreach_get('co', vertices.ravel())
                mesh.loop_triangles.foreach_get('vertices', indices.ravel())
                vcol.data.foreach_get('color', colours.ravel())

            elif vcol.domain == 'CORNER':
                vertices = np.empty((len(mesh.vertices), 3), 'f')
                indices = np.empty((len(mesh.loop_triangles), 3), 'i')
                colours = np.empty((len(vcol.data), 4), 'f')

                mesh.vertices.foreach_get('co', vertices.ravel())
                mesh.loop_triangles.foreach_get('loops', indices.ravel())
                vcol.data.foreach_get('color', colours.ravel())

                vertices = vertices[[l.vertex_index for l in mesh.loops]]

            vbo_format = gpu.types.GPUVertFormat()
            vbo_format.attr_add(
                id='position', comp_type='F32', len=len(vertices[0]), fetch_mode='FLOAT')
            vbo_format.attr_add(
                id='vertex_colour', comp_type='F32', len=len(colours[0]), fetch_mode='FLOAT')

            vbo = gpu.types.GPUVertBuf(vbo_format, len(vertices))
            vbo.attr_fill('position', vertices)
            vbo.attr_fill('vertex_colour', colours)

            ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices)

            batches_matrices.append((
                gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo),
                obj.matrix_world
            ))

    return batches_matrices

# ===========================
# Operators
# ===========================

class SEI_OT_view3d_stencil_visualizer(bpy.types.Operator):
    bl_idname = 'sei.view3d_stencil_visualizer'
    bl_label = 'Visualize Stencil'
    bl_description = 'Toggle the visibility of the stencil for the viewport'

    bl_options = {'REGISTER', 'UNDO'}

    _handle = None

    def _setup_images(self):
        images = {}
        for i in range(MAX_STENCIL_PASSES):
            name = get_image_name(i)
            if bpy.data.images.get(name) is None:
                bpy.data.images.new(name, 8, 8)

            image = bpy.data.images[name]
            image.use_fake_user = True
            image.colorspace_settings.name = 'Non-Color'
            image.pack()
            images[i] = image

        return images

    def _setup_shader(self):
        vsh = \
        '''
        /*
        in vec3 position;
        in vec4 vertex_colour;

        out vec3 vcol;

        uniform mat4 viewproj_matrix;
        uniform mat4 obj_matrix;
        */

        void main()
        {
            gl_Position = viewproj_matrix * obj_matrix * vec4(position, 1.0);
            vcol = vertex_colour.rgb;
        }
        '''

        fsh = \
        '''
        /*
        in vec3 vcol;

        out vec4 col0;
        */

        void main()
        {
            col0 = vec4(vcol, 1.0);
        }
        '''

        shader_info = gpu.types.GPUShaderCreateInfo()

        shader_info.vertex_source(vsh)
        shader_info.fragment_source(fsh)

        # vsh attributes
        shader_info.vertex_in(0, 'VEC3', 'position')
        shader_info.vertex_in(1, 'VEC4', 'vertex_colour')

        interface_info = gpu.types.GPUStageInterfaceInfo("attrs_out")
        interface_info.smooth('VEC3', 'vcol')
        shader_info.vertex_out(interface_info)

        # uniforms
        shader_info.push_constant('MAT4', 'viewproj_matrix')
        shader_info.push_constant('MAT4', 'obj_matrix')

        # write
        shader_info.fragment_out(0, 'VEC4', 'col0')

        return gpu.shader.create_from_info(shader_info)

    def draw_stencil(self, context, shader, images):
        scene = context.scene
        depsgraph = context.evaluated_depsgraph_get()

        pass_collections = get_pass_collections(scene)

        if not pass_collections:
            return

        if context and context.region_data.view_perspective != 'CAMERA':
            _, _, width, height = gpu.state.viewport_get()

            matrix = context.region_data.perspective_matrix

        else:
            width = scene.render.resolution_x
            height = scene.render.resolution_y

            matrix = \
            scene.camera.calc_matrix_camera(depsgraph, x=width, y=height) \
            @ scene.camera.matrix_world.inverted()

        width  = width * scene.render.resolution_percentage // 100
        height = height * scene.render.resolution_percentage // 100

        for pass_idx, collections in pass_collections.items():
            image = images.get(pass_idx)
            if image is None:
                continue

            batches_matrices = build_batches_for_collections(collections, depsgraph)

            if not batches_matrices:
                continue

            offscreen = gpu.types.GPUOffScreen(width, height, format='RGBA8')

            with offscreen.bind():
                framebuffer = gpu.state.active_framebuffer_get()
                framebuffer.clear(depth = 1.0)

                gpu.state.depth_mask_set(True)
                gpu.state.depth_test_set('LESS')
                gpu.state.blend_set('NONE')

                shader.uniform_float('viewproj_matrix', matrix)

                for batch, obj_matrix in batches_matrices:
                    shader.uniform_float('obj_matrix', obj_matrix)
                    batch.draw(shader)

                buffer = framebuffer.read_color(0, 0, width, height, 4, 0, 'FLOAT')
                buffer.dimensions = width * height * 4

            offscreen.free()

            image.scale(width, height)
            image.pixels.foreach_set(buffer)


    @classmethod
    def poll(cls, context):
        return \
        context.area \
        and context.area.type == 'VIEW_3D' \
        and has_any_stencil_objects(context.scene)

    def execute(self, context):
        if SEI_OT_view3d_stencil_visualizer._handle:
            bpy.types.SpaceView3D.draw_handler_remove(
                SEI_OT_view3d_stencil_visualizer._handle,
                'WINDOW'
            )
            SEI_OT_view3d_stencil_visualizer._handle = None

        else:
            images = self._setup_images()
            shader = self._setup_shader()

            SEI_OT_view3d_stencil_visualizer._handle = \
            bpy.types.SpaceView3D.draw_handler_add(
                self.draw_stencil,
                (context, shader, images),
                'WINDOW',
                'POST_PIXEL'
            )

        context.area.tag_redraw()

        return {'FINISHED'}

class SEI_OT_stencil_render(bpy.types.Operator):
    bl_idname = 'sei.stencil_render'
    bl_label = 'Render Stencil'
    bl_description = 'Render active scene with the stencil passes.'

    bl_options = {'REGISTER', 'UNDO'}

    render_image: bpy.props.BoolProperty(name='Render Image')

    @classmethod
    def poll(cls, context):
        return has_any_stencil_objects(context.scene)

    def execute(self, context):
        def print_message(message: str="") -> None:
            if DEBUG_MODE is True:
                print(f'SStencil: {message}')

        #########
        # Initialize values.
        _saved_attrs_render = []
        _saved_attrs_nodes = []

        _tmp_filepaths = {}
        _tmp_images = {}

        scene = context.scene

        #########
        # Group collections by stencil pass.
        print_message('Grouping collections by stencil pass.')

        pass_collections = get_pass_collections(scene)

        if not pass_collections:
            self.report({'WARNING'}, 'No collections assigned to stencil passes.')
            return {'FINISHED'}

        #########
        # Get the image nodes for each pass.
        print_message('Get the image nodes.')

        pass_nodes = {}  # pass_idx -> [nodes]

        # TODO: Only get scene materials.
        for mat in bpy.data.materials:
            stack = [mat.node_tree]

            while stack:
                current_tree = stack.pop()

                if current_tree is None:
                    continue

                for node in current_tree.nodes:
                    if node.type == 'GROUP':
                        stack.append(node.node_tree)

                    elif node.type == 'TEX_IMAGE' \
                    and node.image:
                        for pass_idx in pass_collections:
                            if node.image.name == get_image_name(pass_idx):
                                if pass_idx not in pass_nodes:
                                    pass_nodes[pass_idx] = []
                                pass_nodes[pass_idx].append(node)

        if not pass_nodes:
            self.report({'WARNING'},
                f'No nodes were found with images named "{IMAGE_NAME_PREFIX}_*".')
            return {'FINISHED'}

        # Only render passes that have both collections AND image nodes.
        active_passes = sorted(set(pass_collections.keys()) & set(pass_nodes.keys()))

        if not active_passes:
            self.report({'WARNING'},
                'No stencil passes have both collections and image nodes assigned.')
            return {'FINISHED'}

        #########
        # Get and create the new filepath.
        print_message('Get and create the new filepath.')

        org_filepath = bpy.path.abspath(scene.render.filepath)

        # TODO: bpy.ops.render.render() creates the filepath.
        if os.path.exists(org_filepath) is False:
            os.makedirs(org_filepath, exist_ok=True)

        directory, filename = os.path.split(org_filepath)
        # name, extension = os.path.splitext(filename)

        del org_filepath

        #########
        # Get and set the render attributes.
        print_message('Get and set the render attributes.')

        # NOTE: A dictionary added unecessary complexity.
        attrs_render = [
            (scene.render, 'engine', 'BLENDER_WORKBENCH'),
            (scene.render, 'use_simplify', False),
            (scene.render, 'film_transparent', True),
            (scene.render, 'use_freestyle', False),
            (scene.render, 'use_compositing', False),
            (scene.render, 'use_sequencer', False),
            (scene.render, 'use_file_extension', True),
            (scene.render, 'use_render_cache', False),
            (scene.render, 'use_overwrite', True),
            (scene.render, 'use_placeholder', False),

            (scene.display, 'render_aa', '8'),

            (scene.grease_pencil_settings, 'antialias_threshold', 1.0),

            (scene.display.shading, 'light', 'FLAT'),
            (scene.display.shading, 'color_type', 'VERTEX'),
            (scene.display.shading, 'show_backface_culling', False),
            (scene.display.shading, 'show_xray', False),
            (scene.display.shading, 'show_shadows', False),
            (scene.display.shading, 'show_cavity', False),
            (scene.display.shading, 'use_dof', False),
            (scene.display.shading, 'show_object_outline', False),

            (scene.display_settings, 'display_device', 'sRGB'),

            (scene.render.image_settings.view_settings, 'view_transform', 'Standard'),
            (scene.render.image_settings.view_settings, 'look', 'None'),
            (scene.render.image_settings.view_settings, 'exposure', 0.0),
            (scene.render.image_settings.view_settings, 'gamma', 1.0),
            (scene.render.image_settings.view_settings, 'use_curve_mapping', False),
            (scene.render.image_settings.view_settings, 'use_white_balance', False),

            (scene.render.image_settings, 'file_format', 'OPEN_EXR'), # order matters
            (scene.render.image_settings, 'color_mode', 'RGBA'),
            (scene.render.image_settings, 'color_depth', '16'),
            (scene.render.image_settings, 'exr_codec', 'ZIP')
        ]

        # Save and set.
        for obj, attr, value in attrs_render:
            if hasattr(obj, attr):
                _saved_attrs_render.append((obj, attr, getattr(obj, attr)))
                setattr(obj, attr, value)

        del attrs_render

        # Save filepath separately since it changes per pass.
        _saved_filepath = scene.render.filepath

        #########
        # Render stencil images for each active pass.
        for pass_idx in active_passes:
            collections = pass_collections[pass_idx]
            print_message(f'Rendering stencil pass {pass_idx}.')

            # Unhide the collections for this pass.
            saved_coll_hide = []
            for coll in collections:
                saved_coll_hide.append((coll, coll.hide_render))
                # TODO: Return if collection_layer.exclude.
                coll.hide_render = False

            # Get the set of object names in this pass's collections.
            pass_obj_names = set()
            for coll in collections:
                for obj in coll.all_objects:
                    pass_obj_names.add(obj.name)

            # Get and hide non stencil objects for this pass.
            saved_obj_hide = []
            for obj in scene.objects:
                if obj.type == 'CAMERA' \
                or obj.hide_render is True \
                or obj.name in pass_obj_names:
                    continue

                # Save and set.
                saved_obj_hide.append((obj, obj.hide_render))
                obj.hide_render = True

            # Set filepath for this pass.
            _tmp_filepath = os.path.join(
                directory, f'_tmp_sstencil_{pass_idx}') + os.sep
            os.makedirs(_tmp_filepath, exist_ok=True)
            scene.render.filepath = _tmp_filepath
            _tmp_filepaths[pass_idx] = _tmp_filepath

            # Render (stencil images).
            print_message(f'Rendering the stencil images for pass {pass_idx}.')

            if self.render_image:
                bpy.ops.render.render(write_still=True, use_viewport=True)
            else:
                bpy.ops.render.render(animation=True, use_viewport=True)

            # Restore visibility for this pass.
            for coll, value in saved_coll_hide:
                coll.hide_render = value

            for obj, value in saved_obj_hide:
                obj.hide_render = value

        #########
        # Restore the render attributes.
        print_message('Restoring the attributes.')

        scene.render.filepath = _saved_filepath

        # NOTE: reversed() restores dependant attributes
        # (e.g., PNG vs. OPEN_EXR) in the correct order.
        for obj, attr, value in reversed(_saved_attrs_render):
            if hasattr(obj, attr):
                setattr(obj, attr, value)

        #########
        # Set the image sequences to the image nodes.
        print_message('Setting the image sequences to the image nodes.')

        for pass_idx in active_passes:
            _tmp_filepath = _tmp_filepaths[pass_idx]
            nodes = pass_nodes[pass_idx]

            stencil_frames = os.listdir(_tmp_filepath)
            stencil_frames_length = len(stencil_frames)

            node_filepath = os.path.join(
                _tmp_filepath, sorted(stencil_frames)[0])

            _tmp_image = bpy.data.images.load(node_filepath)
            _tmp_images[pass_idx] = _tmp_image

            for node in nodes:
                # Save and set.
                if hasattr(node, 'image'):
                    _saved_attrs_nodes.append((node, 'image', node.image))
                    node.image = _tmp_image

                if self.render_image:
                    continue

                attrs_nodes = [
                    (node.image.colorspace_settings, 'name', 'Non-Color'),

                    (node.image, 'source', 'SEQUENCE'),

                    (node.image_user, 'frame_duration', stencil_frames_length),
                    (node.image_user, 'frame_start', scene.frame_start),
                    (node.image_user, 'frame_offset', scene.frame_start - 1)
                ]

                # Set.
                for obj, attr, value in attrs_nodes:
                    if hasattr(obj, attr):
                        setattr(obj, attr, value)

        #########
        # Render (normally).
        print_message('Rendering.')

        if self.render_image:
            bpy.ops.render.render(write_still=True, use_viewport=True)
        else:
            bpy.ops.render.render(animation=True, use_viewport=True)

        #########
        # Restore the image nodes attributes.
        print_message('Restoring the image nodes attributes.')

        # NOTE: reversed() restores...
        for obj, attr, value in reversed(_saved_attrs_nodes):
            if hasattr(obj, attr):
                setattr(obj, attr, value)

        #########
        # Delete the temporary files.
        print_message('Deleting the temporary files.')

        for pass_idx, tmp_image in _tmp_images.items():
            bpy.data.images.remove(tmp_image)

        for pass_idx, _tmp_filepath in _tmp_filepaths.items():
            for file in os.listdir(_tmp_filepath):
                filepath = os.path.join(_tmp_filepath, file)

                if os.path.isfile(filepath):
                    os.remove(filepath)

            os.rmdir(_tmp_filepath)

        ###

        self.report({'INFO'}, 'Successfully rendered the image(s).')
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, 'render_image') # RENDER_STILL

class SEI_PT_stencil(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Sei'

    bl_idname = 'SEI_PT_stencil'
    bl_label = 'Stencil Tools'

    def draw_header(self, context):
        self.layout.operator('wm.url_open', text='', icon='HELP').url = 'seilotte.github.io'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene

        # Collection list with add/remove buttons.
        row = layout.row()
        row.template_list(
            'SEI_UL_stencil_collections', '',
            scene, 'stencil_collections',
            scene, 'stencil_collections_active_index',
            rows=3
        )

        col = row.column(align=True)
        col.operator('sei.stencil_collection_add', icon='ADD', text='')
        col.operator('sei.stencil_collection_remove', icon='REMOVE', text='')

        layout.separator()

        col = layout.column()
        col.operator(
            'sei.view3d_stencil_visualizer',
            text = 'Visualize',
            icon = 'PAUSE' if SEI_OT_view3d_stencil_visualizer._handle else 'PLAY'
        )
        col.operator(
            'sei.stencil_render',
            text = 'Render Animation',
            icon = 'RENDER_ANIMATION'
        )

        layout.separator()

        col = layout.column(align=True)
        col.prop(context.scene.render, "resolution_x", text="Resolution X")
        col.prop(context.scene.render, "resolution_y", text="Y")
        col.prop(context.scene.render, "resolution_percentage", text="%")

# ===========================

classes = [
    SEI_PG_stencil_collection,
    SEI_UL_stencil_collections,
    SEI_OT_stencil_collection_add,
    SEI_OT_stencil_collection_remove,
    SEI_OT_view3d_stencil_visualizer,
    SEI_OT_stencil_render,
    SEI_PT_stencil
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.stencil_collections = bpy.props.CollectionProperty(
        type = SEI_PG_stencil_collection,
        name = 'Stencil Collections',
        description = 'Collections assigned to stencil passes'
    )
    bpy.types.Scene.stencil_collections_active_index = bpy.props.IntProperty(
        name = 'Active Stencil Collection Index',
        default = 0
    )

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.stencil_collections
    del bpy.types.Scene.stencil_collections_active_index

if __name__ == "__main__": # debug; live edit
    register()

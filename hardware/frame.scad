// Head-worn eye tracker: parametric glasses frame
//
// Holds one eye camera looking back at the right eye and one scene camera
// looking forward. Written parametrically rather than shipped as an STL so
// that the dimensions that matter can be changed for a different face or a
// different camera without redoing the model.
//
//   openscad -o frame.stl frame.scad
//   openscad -o frame_eye_arm.stl -D part=\"eye_arm\" frame.scad
//
// Print notes are at the bottom of this file.

/* [Which part] */
// "all" previews the assembly; the others are the printable parts.
part = "all";  // [all, front, eye_arm, temple]

/* [Face fit] */
// Pupillary distance. The single most important number here: get it wrong and
// the eye camera looks at a cheekbone.
pupillary_distance = 63;
// Width across the temples, at the hinge.
head_width = 145;
// Bridge width, between the lens openings.
bridge_width = 18;
// How far the front sits from the eyes.
vertex_distance = 14;

/* [Eye camera] */
// A bare USB camera module, without its housing.
eye_cam_w = 8.5;
eye_cam_h = 8.5;
eye_cam_d = 5.0;
// How far below the eye the camera sits, and how far forward.
eye_cam_drop = 22;
eye_cam_forward = 26;
// Angle up towards the eye. 30-35 degrees keeps the iris in frame across the
// full range of gaze without the arm entering the wearer's field of view.
eye_cam_pitch = 32;

/* [Scene camera] */
scene_cam_w = 12;
scene_cam_h = 12;
scene_cam_d = 6;
// Sitting on the bridge keeps it near the cyclopean eye, which minimises the
// parallax between what it sees and what the wearer sees. It cannot remove it.
scene_cam_rise = 6;

/* [Structure] */
front_thickness = 4;
front_height = 34;
lens_opening_w = 46;
lens_opening_h = 28;
temple_length = 130;
temple_thickness = 4;
arm_thickness = 3.5;
arm_width = 7;

/* [Print] */
// Clearance on every camera pocket. 0.3 mm suits a well-tuned FDM printer;
// raise it to 0.4 if parts need forcing, because forcing a camera into a
// pocket is how the lens gets scratched.
fit_clearance = 0.3;

$fn = 48;

// ---------------------------------------------------------------- front

module lens_opening() {
    // Rounded rectangle, built from a hull of four cylinders. Sharp internal
    // corners concentrate stress and this part gets flexed every time the
    // glasses are put on.
    r = 6;
    hull() {
        for (dx = [-1, 1], dy = [-1, 1])
            translate([dx * (lens_opening_w / 2 - r), dy * (lens_opening_h / 2 - r), 0])
                cylinder(r = r, h = front_thickness * 3, center = true);
    }
}

module front() {
    half = pupillary_distance / 2;
    width = pupillary_distance + lens_opening_w + 14;

    difference() {
        // Body
        hull() {
            for (dx = [-1, 1])
                translate([dx * (width / 2 - 8), 0, 0])
                    cylinder(r = 8, h = front_thickness, center = true);
        }

        // The two lens openings
        for (dx = [-1, 1])
            translate([dx * half, 0, 0]) lens_opening();

        // Bridge relief, so the frame does not sit on the nose ridge
        translate([0, -front_height / 2, 0])
            scale([bridge_width / 2, 8, 1])
                cylinder(r = 1, h = front_thickness * 3, center = true);

        // Scene camera pocket, on the bridge
        translate([0, front_height / 2 - scene_cam_h / 2 + scene_cam_rise, 0])
            cube([scene_cam_w + fit_clearance,
                  scene_cam_h + fit_clearance,
                  front_thickness * 3], center = true);
    }
}

// ---------------------------------------------------------------- eye arm

module eye_camera_pocket() {
    cube([eye_cam_w + fit_clearance,
          eye_cam_h + fit_clearance,
          eye_cam_d + fit_clearance], center = true);
}

module eye_arm() {
    // An L: down from the frame, then forward and up under the eye.
    difference() {
        union() {
            // Vertical leg
            translate([0, -eye_cam_drop / 2, 0])
                cube([arm_width, eye_cam_drop, arm_thickness], center = true);

            // Forward leg, pitched up towards the eye
            translate([0, -eye_cam_drop, 0])
                rotate([eye_cam_pitch, 0, 0])
                    translate([0, 0, eye_cam_forward / 2])
                        cube([arm_width, arm_thickness, eye_cam_forward], center = true);

            // Camera boss at the end
            translate([0, -eye_cam_drop, 0])
                rotate([eye_cam_pitch, 0, 0])
                    translate([0, 0, eye_cam_forward])
                        cube([eye_cam_w + 4, eye_cam_h + 4, eye_cam_d + 3], center = true);
        }

        // Camera pocket, opening forwards
        translate([0, -eye_cam_drop, 0])
            rotate([eye_cam_pitch, 0, 0])
                translate([0, 0, eye_cam_forward])
                    eye_camera_pocket();

        // Cable channel down the back of the arm
        translate([0, -eye_cam_drop / 2, -arm_thickness / 2])
            cube([2.5, eye_cam_drop + 2, 1.5], center = true);
    }
}

// ---------------------------------------------------------------- temple

module temple() {
    difference() {
        union() {
            cube([temple_thickness, temple_length, 9], center = true);

            // Ear hook
            translate([0, temple_length / 2, 0])
                rotate([0, 90, 0])
                    rotate_extrude(angle = 70, $fn = 64)
                        translate([16, 0, 0])
                            square([temple_thickness, 9], center = true);
        }

        // Hinge pin hole
        translate([0, -temple_length / 2 + 6, 0])
            rotate([0, 90, 0])
                cylinder(r = 1.2, h = temple_thickness * 3, center = true);
    }
}

// ---------------------------------------------------------------- assembly

module assembly() {
    color("LightSteelBlue") front();

    // Eye arm on the right eye only. One eye is enough: gaze is conjugate for
    // anything beyond arm's length, and a second eye camera doubles the
    // weight, the cabling and the synchronisation problem for very little.
    translate([pupillary_distance / 2, -front_height / 2 + 4, vertex_distance / 2])
        color("Tomato") eye_arm();

    for (dx = [-1, 1])
        translate([dx * head_width / 2, front_height / 2 - temple_length / 2, 0])
            color("Gainsboro") temple();
}

if (part == "all") assembly();
else if (part == "front") front();
else if (part == "eye_arm") eye_arm();
else if (part == "temple") temple();

// ---------------------------------------------------------------- printing
//
// PETG rather than PLA. The frame sits on a face for an hour at a time and PLA
// softens enough at skin temperature over that period to let the eye arm
// droop, which shows up as a slow drift in gaze that looks exactly like
// calibration decay and is not.
//
// Front:    flat on the bed, no supports.
// Eye arm:  on its side, supports on. This part carries the camera and any
//           layer delamination in it becomes gaze error, so 4 perimeters.
// Temple:   flat, no supports; the ear hook bridges fine at this radius.
//
// After printing, check the eye arm does not enter the wearer's field of view
// when they look down. If it does, reduce eye_cam_pitch before reducing
// eye_cam_forward -- moving the camera closer costs more accuracy than
// flattening its angle.

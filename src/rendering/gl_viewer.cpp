#include "rendering/gl_viewer.h"
#include <iostream>
#include <algorithm>

namespace ar_slam {

// Vertex shader
const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = 5.0;
    vertexColor = aColor;
}
)";

// Fragment shader
const char* fragment_shader_source = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    // Make points circular
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(length(coord) > 0.5)
        discard;

    FragColor = vec4(vertexColor, 1.0);
}
)";

GLViewer::GLViewer(const std::string& title) : window_(nullptr) {
}

GLViewer::~GLViewer() {
    cleanup();
}

bool GLViewer::init() {
    if (!init_glfw()) return false;
    if (!init_opengl()) return false;

    // Set up initial view
    projection_matrix_ = glm::perspective(glm::radians(45.0f),
                                         (float)width_ / height_,
                                         0.1f, 100.0f);
    update_view_matrix();

    return true;
}

bool GLViewer::init_glfw() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width_, height_, "AR SLAM 3D Viewer", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    return true;
}

bool GLViewer::init_opengl() {
    // Compile shaders
    shader_program_ = compile_shaders();
    if (shader_program_ == 0) return false;

    // Create VAO and VBOs
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_points_);
    glGenBuffers(1, &vbo_colors_);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Set background color
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);

    return true;
}

GLuint GLViewer::compile_shaders() {
    // Compile vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    glCompileShader(vertex_shader);

    // Check compilation
    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertex_shader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        return 0;
    }

    // Compile fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    glCompileShader(fragment_shader);

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        return 0;
    }

    // Link program
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed: " << infoLog << std::endl;
        return 0;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

void GLViewer::update_view_matrix() {
    glm::vec3 eye(
        camera_distance_ * sin(camera_angle_y_) * cos(camera_angle_x_),
        camera_distance_ * sin(camera_angle_x_),
        camera_distance_ * cos(camera_angle_y_) * cos(camera_angle_x_)
    );

    view_matrix_ = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

void GLViewer::update_points(const std::vector<cv::Point3f>& points) {
    points_.clear();
    colors_.clear();

    // Find min/max Z for color mapping
    float min_z = 100.0f, max_z = -100.0f;
    for (const auto& p : points) {
        min_z = std::min(min_z, p.z);
        max_z = std::max(max_z, p.z);
    }
    float z_range = max_z - min_z + 0.001f;  // Avoid division by zero

    for (const auto& p : points) {
        points_.push_back(glm::vec3(p.x, p.y, p.z));

        // Color based on depth: close=red, mid=green, far=blue
        float t = (p.z - min_z) / z_range;

        if (t < 0.5f) {
            // Red to Green
            float s = t * 2.0f;
            colors_.push_back(glm::vec3(1.0f - s, s, 0.0f));
        } else {
            // Green to Blue
            float s = (t - 0.5f) * 2.0f;
            colors_.push_back(glm::vec3(0.0f, 1.0f - s, s));
        }
    }
}

void GLViewer::process_input() {
    // Camera rotation
    if (glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera_angle_y_ -= 0.02f;
    if (glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera_angle_y_ += 0.02f;
    if (glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS)
        camera_distance_ -= 0.1f;
    if (glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera_distance_ += 0.1f;

    // Clamp camera distance
    if (camera_distance_ < 1.0f) camera_distance_ = 1.0f;
    if (camera_distance_ > 20.0f) camera_distance_ = 20.0f;

    // Exit on ESC
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);
}

void GLViewer::draw_axes() {
    // Draw coordinate axes using lines
    // This is a simplified version - in production you'd use a separate shader

    // Store current point size
    GLfloat pointSize;
    glGetFloatv(GL_POINT_SIZE, &pointSize);

    // Draw thicker lines for axes
    glLineWidth(3.0f);

    // We'll add 6 points for the axes (2 per axis)
    std::vector<glm::vec3> axis_points = {
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0),  // X axis
        glm::vec3(0, 0, 0), glm::vec3(0, 1, 0),  // Y axis
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 1)   // Z axis
    };

    std::vector<glm::vec3> axis_colors = {
        glm::vec3(1, 0, 0), glm::vec3(1, 0, 0),  // Red for X
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0),  // Green for Y
        glm::vec3(0, 0, 1), glm::vec3(0, 0, 1)   // Blue for Z
    };

    // Draw axes
    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_points_);
    glBufferData(GL_ARRAY_BUFFER, axis_points.size() * sizeof(glm::vec3),
                axis_points.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
    glBufferData(GL_ARRAY_BUFFER, axis_colors.size() * sizeof(glm::vec3),
                axis_colors.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glDrawArrays(GL_LINES, 0, 6);

    glBindVertexArray(0);

    // Restore line width
    glLineWidth(1.0f);
}

bool GLViewer::render() {
    if (glfwWindowShouldClose(window_)) return false;

    // Process input
    process_input();
    update_view_matrix();

    // Clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use shader
    glUseProgram(shader_program_);

    // Set uniforms
    GLuint view_loc = glGetUniformLocation(shader_program_, "view");
    GLuint proj_loc = glGetUniformLocation(shader_program_, "projection");
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, &view_matrix_[0][0]);
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, &projection_matrix_[0][0]);

    // Draw coordinate axes first
    draw_axes();

    // Draw points if we have any
    if (!points_.empty()) {
        glBindVertexArray(vao_);

        // Update point buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo_points_);
        glBufferData(GL_ARRAY_BUFFER, points_.size() * sizeof(glm::vec3),
                    points_.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        // Update color buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(glm::vec3),
                    colors_.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        // Draw
        glDrawArrays(GL_POINTS, 0, points_.size());

        glBindVertexArray(0);
    }

    glfwSwapBuffers(window_);
    glfwPollEvents();

    return true;
}

bool GLViewer::should_close() const {
    return glfwWindowShouldClose(window_);
}

void GLViewer::set_title(const std::string& title) {
    if (window_) {
        glfwSetWindowTitle(window_, title.c_str());
    }
}

void GLViewer::cleanup() {
    if (shader_program_) glDeleteProgram(shader_program_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_points_) glDeleteBuffers(1, &vbo_points_);
    if (vbo_colors_) glDeleteBuffers(1, &vbo_colors_);

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

} // namespace ar_slam
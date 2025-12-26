#include "rendering/gl_viewer.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ar_slam {

// Vertex shader source
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

// Fragment shader source
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

GLViewer::GLViewer(const std::string& title)
    : window_(nullptr), window_title_(title) {
    // Constructor just initializes members
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

    // Use the stored window title
    window_ = glfwCreateWindow(width_, height_, window_title_.c_str(), nullptr, nullptr);
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

    // Create Grid VBOs
    glGenBuffers(1, &vbo_grid_points_);
    glGenBuffers(1, &vbo_grid_colors_);

    // Initialize Grid geometry
    init_grid();

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

    // Check vertex shader compilation
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

    // Check fragment shader compilation
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        return 0;
    }

    // Create and link shader program
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    // Check linking
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed: " << infoLog << std::endl;
        return 0;
    }

    // Clean up individual shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

void GLViewer::update_view_matrix() {
    // Calculate eye position based on spherical coordinates
    glm::vec3 eye(
        camera_distance_ * sin(camera_angle_y_) * cos(camera_angle_x_),
        camera_distance_ * sin(camera_angle_x_),
        camera_distance_ * cos(camera_angle_y_) * cos(camera_angle_x_)
    );

    // Look at origin with Y-up
    view_matrix_ = glm::lookAt(eye, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

void GLViewer::update_points(const std::vector<cv::Point3f>& points) {
    points_.clear();
    colors_.clear();

    if (points.empty()) return;

    // Find min/max Z for color mapping
    float min_z = points[0].z;
    float max_z = points[0].z;

    for (const auto& p : points) {
        min_z = std::min(min_z, p.z);
        max_z = std::max(max_z, p.z);
    }

    float z_range = max_z - min_z;
    if (z_range < 0.001f) z_range = 1.0f;  // Avoid division by zero

    // Convert points and assign colors based on depth
    for (const auto& p : points) {
        points_.push_back(glm::vec3(p.x, p.y, p.z));

        // Color gradient: near (red) -> mid (green) -> far (blue)
        float t = (p.z - min_z) / z_range;

        glm::vec3 color;
        if (t < 0.5f) {
            // Red to Green transition
            float s = t * 2.0f;
            color = glm::vec3(1.0f - s, s, 0.0f);
        } else {
            // Green to Blue transition
            float s = (t - 0.5f) * 2.0f;
            color = glm::vec3(0.0f, 1.0f - s, s);
        }
        colors_.push_back(color);
    }
}

void GLViewer::update_points_with_colors(const std::vector<cv::Point3f>& points,
                                         const std::vector<cv::Vec3b>& colors) {
    points_.clear();
    colors_.clear();

    size_t num_points = std::min(points.size(), colors.size());

    for (size_t i = 0; i < num_points; ++i) {
        points_.push_back(glm::vec3(points[i].x, points[i].y, points[i].z));
        colors_.push_back(glm::vec3(colors[i][2] / 255.0f,  // BGR to RGB
                                    colors[i][1] / 255.0f,
                                    colors[i][0] / 255.0f));
    }
}

void GLViewer::process_input() {
    // Camera rotation with arrow keys
    if (glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera_angle_y_ -= 0.02f;
    if (glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera_angle_y_ += 0.02f;
    if (glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS)
        camera_distance_ -= 0.1f;
    if (glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera_distance_ += 0.1f;

    // Clamp camera distance
    camera_distance_ = std::max(1.0f, std::min(camera_distance_, 20.0f));

    // Exit on ESC
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);
}

void GLViewer::draw_axes() {
    // Prepare axis vertices and colors
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

    // Set line width for axes
    glLineWidth(3.0f);

    // Bind VAO and update buffers
    glBindVertexArray(vao_);

    // Update vertex positions
    glBindBuffer(GL_ARRAY_BUFFER, vbo_points_);
    glBufferData(GL_ARRAY_BUFFER, axis_points.size() * sizeof(glm::vec3),
                axis_points.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // Update vertex colors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
    glBufferData(GL_ARRAY_BUFFER, axis_colors.size() * sizeof(glm::vec3),
                axis_colors.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    // Draw the axes
    glDrawArrays(GL_LINES, 0, 6);

    glBindVertexArray(0);

    // Reset line width
    glLineWidth(1.0f);
}

void GLViewer::init_grid() {
    std::vector<glm::vec3> grid_points;
    std::vector<glm::vec3> grid_colors;

    float size = 10.0f;
    float step = 1.0f;
    glm::vec3 color(0.3f, 0.3f, 0.3f); // Dark gray grid lines

    for (float i = -size; i <= size; i += step) {
        // Lines parallel to X
        grid_points.push_back(glm::vec3(-size, -2.0f, i));
        grid_points.push_back(glm::vec3(size, -2.0f, i));
        grid_colors.push_back(color);
        grid_colors.push_back(color);

        // Lines parallel to Z
        grid_points.push_back(glm::vec3(i, -2.0f, -size));
        grid_points.push_back(glm::vec3(i, -2.0f, size));
        grid_colors.push_back(color);
        grid_colors.push_back(color);
    }

    grid_vertex_count_ = grid_points.size();

    // Upload grid data to static buffers
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid_points_);
    glBufferData(GL_ARRAY_BUFFER, grid_points.size() * sizeof(glm::vec3),
                grid_points.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid_colors_);
    glBufferData(GL_ARRAY_BUFFER, grid_colors.size() * sizeof(glm::vec3),
                grid_colors.data(), GL_STATIC_DRAW);
}

void GLViewer::draw_grid() {
    if (grid_vertex_count_ == 0) return;

    glLineWidth(1.0f);
    glBindVertexArray(vao_);

    // Bind grid buffers
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid_points_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid_colors_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    // Draw grid
    glDrawArrays(GL_LINES, 0, grid_vertex_count_);

    glBindVertexArray(0);
}

bool GLViewer::render() {
    if (glfwWindowShouldClose(window_)) return false;

    // Process keyboard input
    process_input();

    // Update camera view matrix
    update_view_matrix();

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate shader program
    glUseProgram(shader_program_);

    // Set transformation matrices
    GLuint view_loc = glGetUniformLocation(shader_program_, "view");
    GLuint proj_loc = glGetUniformLocation(shader_program_, "projection");
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, &view_matrix_[0][0]);
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, &projection_matrix_[0][0]);

    // Draw grid
    draw_grid();

    // Draw coordinate axes
    draw_axes();

    // Draw point cloud if available
    if (!points_.empty()) {
        glBindVertexArray(vao_);

        // Update vertex positions
        glBindBuffer(GL_ARRAY_BUFFER, vbo_points_);
        glBufferData(GL_ARRAY_BUFFER, points_.size() * sizeof(glm::vec3),
                    points_.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        // Update vertex colors
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(glm::vec3),
                    colors_.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        // Draw points
        glDrawArrays(GL_POINTS, 0, points_.size());

        glBindVertexArray(0);
    }

    // Swap buffers and poll events
    glfwSwapBuffers(window_);
    glfwPollEvents();

    return true;
}

bool GLViewer::should_close() const {
    return glfwWindowShouldClose(window_);
}

void GLViewer::set_title(const std::string& title) {
    window_title_ = title;
    if (window_) {
        glfwSetWindowTitle(window_, window_title_.c_str());
    }
}

void GLViewer::mouse_callback(double xpos, double ypos) {
    // Suppress unused parameter warnings
    (void)xpos;
    (void)ypos;
    // TODO: Implement mouse-based camera control
}

void GLViewer::scroll_callback(double yoffset) {
    // Zoom with scroll
    camera_distance_ -= static_cast<float>(yoffset) * 0.5f;
    camera_distance_ = std::max(1.0f, std::min(camera_distance_, 20.0f));
}

void GLViewer::cleanup() {
    // Delete OpenGL objects
    if (shader_program_) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }

    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }

    if (vbo_points_) {
        glDeleteBuffers(1, &vbo_points_);
        vbo_points_ = 0;
    }

    if (vbo_colors_) {
        glDeleteBuffers(1, &vbo_colors_);
        vbo_colors_ = 0;
    }

    if (vbo_grid_points_) {
        glDeleteBuffers(1, &vbo_grid_points_);
        vbo_grid_points_ = 0;
    }

    if (vbo_grid_colors_) {
        glDeleteBuffers(1, &vbo_grid_colors_);
        vbo_grid_colors_ = 0;
    }

    // Destroy window and terminate GLFW
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
}

} // namespace ar_slam
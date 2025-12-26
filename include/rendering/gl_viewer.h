#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace ar_slam {

/**
 * @brief OpenGL-based 3D viewer for point cloud visualization
 *
 * This class provides real-time 3D visualization of tracked features
 * with depth-based coloring and interactive camera controls.
 */
class GLViewer {
public:
    /**
     * @brief Constructor
     * @param title Window title for the viewer
     */
    explicit GLViewer(const std::string& title = "AR SLAM 3D Viewer");

    /**
     * @brief Destructor - cleans up OpenGL resources
     */
    ~GLViewer();

    /**
     * @brief Initialize the viewer (GLFW window and OpenGL context)
     * @return true if initialization successful, false otherwise
     */
    bool init();

    /**
     * @brief Clean up all resources
     */
    void cleanup();

    /**
     * @brief Update the point cloud with automatic depth-based coloring
     * @param points Vector of 3D points to display
     */
    void update_points(const std::vector<cv::Point3f>& points);

    /**
     * @brief Update the point cloud with custom colors
     * @param points Vector of 3D points
     * @param colors Vector of RGB colors (CV_8UC3 format)
     */
    void update_points_with_colors(const std::vector<cv::Point3f>& points,
                                   const std::vector<cv::Vec3b>& colors);

    /**
     * @brief Render one frame
     * @return true if rendering successful, false if window should close
     */
    bool render();

    /**
     * @brief Check if the window should close
     * @return true if window close requested
     */
    bool should_close() const;

    /**
     * @brief Set the window title
     * @param title New window title
     */
    void set_title(const std::string& title);

    /**
     * @brief Mouse movement callback (for future mouse control)
     * @param xpos Mouse X position
     * @param ypos Mouse Y position
     */
    void mouse_callback(double xpos, double ypos);

    /**
     * @brief Mouse scroll callback for zoom
     * @param yoffset Scroll offset
     */
    void scroll_callback(double yoffset);

    /**
     * @brief Process keyboard input
     */
    void process_input();

private:
    // Window management (order matters for initialization)
    GLFWwindow* window_;                ///< GLFW window handle
    std::string window_title_;          ///< Window title
    int width_ = 1280;                  ///< Window width
    int height_ = 720;                  ///< Window height

    // Camera parameters
    float camera_distance_ = 5.0f;      ///< Distance from origin
    float camera_angle_x_ = 0.0f;       ///< Vertical rotation angle
    float camera_angle_y_ = 0.0f;       ///< Horizontal rotation angle
    glm::mat4 view_matrix_;            ///< View transformation matrix
    glm::mat4 projection_matrix_;      ///< Projection matrix

    // Point cloud data
    std::vector<glm::vec3> points_;    ///< 3D point positions
    std::vector<glm::vec3> colors_;    ///< Point colors (RGB)

    // OpenGL objects
    GLuint vao_ = 0;                   ///< Vertex Array Object
    GLuint vbo_points_ = 0;            ///< Vertex Buffer Object for positions
    GLuint vbo_colors_ = 0;            ///< Vertex Buffer Object for colors
    GLuint shader_program_ = 0;        ///< Shader program handle

    // Grid resources
    GLuint vbo_grid_points_ = 0;
    GLuint vbo_grid_colors_ = 0;
    int grid_vertex_count_ = 0;
    void init_grid();

    // Mouse control state
    bool mouse_pressed_ = false;       ///< Is mouse button pressed
    double last_mouse_x_ = 0.0;        ///< Last mouse X position
    double last_mouse_y_ = 0.0;        ///< Last mouse Y position

    // Private initialization methods
    /**
     * @brief Initialize GLFW and create window
     * @return true if successful
     */
    bool init_glfw();

    /**
     * @brief Initialize OpenGL context and resources
     * @return true if successful
     */
    bool init_opengl();

    /**
     * @brief Compile and link shaders
     * @return Shader program handle, or 0 on failure
     */
    GLuint compile_shaders();

    /**
     * @brief Update the view matrix based on camera parameters
     */
    void update_view_matrix();

    /**
     * @brief Draw coordinate axes
     */
    void draw_axes();

    /**
     * @brief Draw grid floor
     */
    void draw_grid();
};

} // namespace ar_slam
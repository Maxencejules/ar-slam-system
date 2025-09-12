#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace ar_slam {

    class GLViewer {
    private:
        GLFWwindow* window_;
        int width_ = 1280;
        int height_ = 720;

        // Camera controls
        float camera_distance_ = 5.0f;
        float camera_angle_x_ = 0.0f;
        float camera_angle_y_ = 0.0f;
        glm::mat4 view_matrix_;
        glm::mat4 projection_matrix_;

        // Points to render
        std::vector<glm::vec3> points_;
        std::vector<glm::vec3> colors_;

        // OpenGL objects
        GLuint vao_, vbo_points_, vbo_colors_;
        GLuint shader_program_;

        // Mouse control
        bool mouse_pressed_ = false;
        double last_mouse_x_ = 0;
        double last_mouse_y_ = 0;

    public:
        GLViewer(const std::string& title = "AR SLAM 3D Viewer");
        ~GLViewer();

        bool init();
        void cleanup();

        // Update points
        void update_points(const std::vector<cv::Point3f>& points);
        void update_points_with_colors(const std::vector<cv::Point3f>& points,
                                       const std::vector<cv::Vec3b>& colors);

        // Render
        bool render();
        bool should_close() const;

        // Camera control
        void process_input();
        void mouse_callback(double xpos, double ypos);
        void scroll_callback(double yoffset);

        // Window management
        void set_title(const std::string& title);

    private:
        bool init_glfw();
        bool init_opengl();
        GLuint compile_shaders();
        void update_view_matrix();
        void draw_axes();
    };

} // namespace ar_slam
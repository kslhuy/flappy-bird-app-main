import pygame
import random
from ai_controllers import create_controller
from ai_utils import GameStateManager, ImitationLogger, AIConfig, load_residual_model
from gui_controls import ControlPanel

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

# Screen dimensions
GAME_WIDTH = 400
GAME_HEIGHT = 600
GUI_WIDTH = 320
SCREEN_WIDTH = GAME_WIDTH + GUI_WIDTH
SCREEN_HEIGHT = max(GAME_HEIGHT, 650)  # Increased height for better UI spacing
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird with AI Controls")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 235)

# Sprite sheet loading function with bounds checking


def load_sprite_from_sheet(sheet, x, y, width, height):
    sheet_width, sheet_height = sheet.get_size()
    if x + width > sheet_width or y + height > sheet_height or x < 0 or y < 0:
        print(
            f"Error: Sprite at ({x}, {y}, {width}, {height}) exceeds sheet size ({sheet_width}, {sheet_height})")
        return None
    try:
        sprite = sheet.subsurface((x, y, width, height)).convert_alpha()
        print(f"Loaded sprite at ({x}, {y}) with size ({width}, {height})")
        return sprite
    except pygame.error as e:
        print(f"Error loading sprite at ({x}, {y}): {e}")
        return None


# Load the entire sprite sheet
try:
    sprite_sheet = pygame.image.load("spritesheet.png").convert_alpha()
    sheet_width, sheet_height = sprite_sheet.get_size()
    print(f"Sprite sheet loaded: {sheet_width}x{sheet_height}")
except pygame.error as e:
    print(f"Error loading sprite sheet: {e}")
    sprite_sheet = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    sprite_sheet.fill(SKY_BLUE)
    sheet_width, sheet_height = SCREEN_WIDTH, SCREEN_HEIGHT

# Load sound effects (MP3 format)
try:
    die_sound = pygame.mixer.Sound("die.mp3")
    hit_sound = pygame.mixer.Sound("hit.mp3")
    flap_sound = pygame.mixer.Sound("flap.mp3")
    point_sound = pygame.mixer.Sound("point.mp3")
except pygame.error as e:
    print(f"Error loading sound effects: {e}")
    die_sound = hit_sound = flap_sound = point_sound = None

# Define sprite coordinates and sizes
BACKGROUND_X, BACKGROUND_Y, BACKGROUND_WIDTH, BACKGROUND_HEIGHT = 0, 0, 225, 400
PIPE_X, PIPE_Y, PIPE_WIDTH, PIPE_HEIGHT = 0, 504, 43, 250
BIRD_WIDTH, BIRD_HEIGHT = 34, 24
BIRD_FRAMES = [(0, 766), (44, 766), (88, 766)]

# Load background
background = load_sprite_from_sheet(
    sprite_sheet, BACKGROUND_X, BACKGROUND_Y, BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
if background:
    background = pygame.transform.scale(
        background, (GAME_WIDTH, GAME_HEIGHT))
else:
    print("Background failed to load, using fallback")
    background = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
    background.fill(SKY_BLUE)

# Load bird frames
bird_frames = []
for i, (x, y) in enumerate(BIRD_FRAMES):
    frame = load_sprite_from_sheet(sprite_sheet, x, y, BIRD_WIDTH, BIRD_HEIGHT)
    if frame:
        bird_frames.append(frame)
    else:
        print(f"Bird frame {i} failed to load, using fallback")
        fallback = pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT))
        fallback.fill(WHITE)
        bird_frames.append(fallback)

# Load pipe
pipe_base_img = load_sprite_from_sheet(
    sprite_sheet, PIPE_X, PIPE_Y, PIPE_WIDTH, PIPE_HEIGHT)
if not pipe_base_img:
    print("Pipe failed to load, using fallback")
    pipe_base_img = pygame.Surface((PIPE_WIDTH, PIPE_HEIGHT))
    pipe_base_img.fill((0, 255, 0))

# Bird properties
bird_x = GAME_WIDTH // 4
bird_y = GAME_HEIGHT // 2
bird_velocity = 0
GRAVITY = 0.5
FLAP_POWER = -10
bird_frame = 0
bird_animation_speed = 0.1
bird_animation_counter = 0

# Pipe properties
PIPE_GAP = 150
pipe_x = GAME_WIDTH
pipe_height = random.randint(100, GAME_HEIGHT - PIPE_GAP - 100)
pipe_speed = 3
pipe_passed = False

# Game variables
score = 0
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 20)
clock = pygame.time.Clock()

# Game states
IDLE = 0
PLAYING = 1
GAME_OVER = 2
game_state = IDLE

# Auto-replay timer
auto_replay_timer = 0
AUTO_REPLAY_DELAY = 120  # 2 seconds at 60 FPS

# -------------------- AI System Setup --------------------
ai_config = AIConfig()
state_manager = GameStateManager()
logger = ImitationLogger(ai_config.log_path, ai_config.data_log)
residual_model = load_residual_model(ai_config.residual_model_path)

# Initialize AI controllers
controllers = {
    'heuristic': create_controller('heuristic', residual_model=residual_model),
    'pid': create_controller('pid', **ai_config.pid_params),
    'plan': create_controller('plan', **ai_config.planner_params, gravity=GRAVITY, flap_power=FLAP_POWER)
}

# Initialize GUI Control Panel (positioned on the right side)
control_panel = ControlPanel(
    x=GAME_WIDTH + 10, y=10, width=GUI_WIDTH - 20, height=SCREEN_HEIGHT - 20,
    controllers=controllers, ai_config=ai_config
)

def ai_decide():
    """Main AI decision function"""
    state = state_manager.get_state(bird_y, bird_velocity, pipe_height, PIPE_GAP, pipe_x, bird_x)
    controller = controllers[ai_config.ai_mode]
    
    # Update controller parameters from GUI
    if ai_config.ai_mode == 'heuristic':
        heuristic_params = control_panel.get_heuristic_params()
        controller.set_params(**heuristic_params)
        action = controller.decide(state, BIRD_HEIGHT, GAME_WIDTH, GAME_HEIGHT)
    elif ai_config.ai_mode == 'pid':
        pid_params = control_panel.get_pid_params()
        controller.set_params(**pid_params)
        action = controller.decide(state, BIRD_HEIGHT)
    elif ai_config.ai_mode == 'plan':
        planner_params = control_panel.get_planner_params()
        controller.set_params(**planner_params)
        action = controller.decide(state, BIRD_HEIGHT)
    else:
        action = 0
    
    logger.log_sample(state, action)
    return action


def draw_bird(x, y, frame):
    screen.blit(bird_frames[frame], (x, y))


def draw_pipe(x, height):
    top_pipe_height = height
    top_pipe = pygame.transform.flip(pipe_base_img, False, True)
    top_pipe = pygame.transform.scale(top_pipe, (PIPE_WIDTH, top_pipe_height))
    screen.blit(top_pipe, (x, 0))

    bottom_pipe_height = GAME_HEIGHT - (height + PIPE_GAP)
    bottom_pipe = pygame.transform.scale(
        pipe_base_img, (PIPE_WIDTH, bottom_pipe_height))
    screen.blit(bottom_pipe, (x, height + PIPE_GAP))


def check_collision(bird_x, bird_y, pipe_x, pipe_height):
    bird_rect = pygame.Rect(bird_x, bird_y, BIRD_WIDTH, BIRD_HEIGHT)
    top_pipe_rect = pygame.Rect(pipe_x, 0, PIPE_WIDTH, pipe_height)
    bottom_pipe_rect = pygame.Rect(
        pipe_x, pipe_height + PIPE_GAP, PIPE_WIDTH, GAME_HEIGHT - (pipe_height + PIPE_GAP))
    if bird_y < 0 or bird_y + BIRD_HEIGHT > GAME_HEIGHT:
        if die_sound and (bird_y < 0 or bird_y + BIRD_HEIGHT > GAME_HEIGHT):
            die_sound.play()
        return True
    if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
        if hit_sound:
            hit_sound.play()
        return True
    return False


def reset_game():
    global bird_y, bird_velocity, pipe_x, pipe_height, score, pipe_passed, bird_frame, auto_replay_timer
    bird_y = GAME_HEIGHT // 2
    bird_velocity = 0
    pipe_x = GAME_WIDTH
    pipe_height = random.randint(100, GAME_HEIGHT - PIPE_GAP - 100)
    score = 0
    pipe_passed = False
    bird_frame = 0
    auto_replay_timer = 0
    # Reset PID controller state
    if 'pid' in controllers:
        controllers['pid'].reset()


def render_text_with_outline(text, font, color, outline_color):
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, outline_color)
    surface = pygame.Surface(
        (text_surface.get_width() + 4, text_surface.get_height() + 4), pygame.SRCALPHA)
    for dx, dy in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]:
        surface.blit(outline_surface, (dx + 2, dy + 2))
    surface.blit(text_surface, (2, 2))
    return surface


def draw_ai_status():
    status = f"AI: {'ON' if ai_config.use_ai else 'OFF'}"
    if ai_config.use_ai:
        status += f" ({ai_config.ai_mode}) - AUTO"
    txt = render_text_with_outline(status, small_font, WHITE, BLACK)
    rect = txt.get_rect(topright=(GAME_WIDTH - 8, 8))
    screen.blit(txt, rect)
    
    # Show keyboard shortcuts
    shortcuts = [
        "A: Toggle AI",
        "1-3: AI Mode",
        "SPACE: Manual Control",
        "Q: Quit"
    ]
    
    for i, shortcut in enumerate(shortcuts):
        shortcut_txt = render_text_with_outline(shortcut, small_font, WHITE, BLACK)
        shortcut_rect = shortcut_txt.get_rect(topright=(GAME_WIDTH - 8, 35 + i * 15))
        screen.blit(shortcut_txt, shortcut_rect)


# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Let GUI handle events first
        control_panel.handle_event(event)
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if game_state == IDLE:
                    reset_game()
                    game_state = PLAYING
                elif game_state == PLAYING:
                    bird_velocity = FLAP_POWER
                    if flap_sound:
                        flap_sound.play()
                elif game_state == GAME_OVER:
                    reset_game()
                    game_state = IDLE
            elif event.key == pygame.K_q and game_state in (IDLE, GAME_OVER):
                running = False
            elif event.key == pygame.K_a and game_state in (IDLE, GAME_OVER):
                # Toggle AI on/off at idle or after game over
                ai_config.toggle_ai()
                print(f"AI toggled: {ai_config.use_ai}")
                # Auto-start game if AI is enabled and we're in IDLE state
                if ai_config.use_ai and game_state == IDLE:
                    reset_game()
                    game_state = PLAYING
            elif event.key == pygame.K_1:
                ai_config.set_mode('heuristic')
                print('AI mode: heuristic')
                # Auto-start game if AI is enabled and we're in IDLE state
                if ai_config.use_ai and game_state == IDLE:
                    reset_game()
                    game_state = PLAYING
            elif event.key == pygame.K_2:
                ai_config.set_mode('pid')
                print('AI mode: pid')
                # Auto-start game if AI is enabled and we're in IDLE state
                if ai_config.use_ai and game_state == IDLE:
                    reset_game()
                    game_state = PLAYING
            elif event.key == pygame.K_3:
                ai_config.set_mode('plan')
                print('AI mode: plan')
                # Auto-start game if AI is enabled and we're in IDLE state
                if ai_config.use_ai and game_state == IDLE:
                    reset_game()
                    game_state = PLAYING

    if game_state == IDLE:
        # Auto-start game if AI is enabled
        if ai_config.use_ai:
            reset_game()
            game_state = PLAYING
        else:
            # Clear screen and draw background for game area
            screen.fill(BLACK)
            screen.blit(background, (0, 0))
            bird_y = GAME_HEIGHT // 2 + 20 * \
                pygame.math.Vector2(0, 1).rotate(pygame.time.get_ticks() / 100).y
            draw_bird(bird_x, bird_y, int(bird_frame))
            bird_animation_counter += bird_animation_speed
            if bird_animation_counter >= 1:
                bird_animation_counter = 0
                bird_frame = (bird_frame + 1) % len(bird_frames)
            start_text = render_text_with_outline(
                "Press SPACE to Start (Manual Mode)", font, WHITE, BLACK)
            start_rect = start_text.get_rect(
                center=(GAME_WIDTH // 2, GAME_HEIGHT // 2))
            screen.blit(start_text, start_rect)
            
            # Show AI instruction
            ai_text = render_text_with_outline(
                "Press A to enable AI Auto-Play", small_font, WHITE, BLACK)
            ai_rect = ai_text.get_rect(
                center=(GAME_WIDTH // 2, GAME_HEIGHT // 2 + 30))
            screen.blit(ai_text, ai_rect)

    elif game_state == PLAYING:
        # AI control injection
        if ai_config.use_ai:
            action = ai_decide()
            if action:
                bird_velocity = FLAP_POWER
                if flap_sound:
                    flap_sound.play()
        bird_velocity += GRAVITY
        bird_y += bird_velocity
        pipe_x -= pipe_speed

        bird_animation_counter += bird_animation_speed
        if bird_animation_counter >= 1:
            bird_animation_counter = 0
            bird_frame = (bird_frame + 1) % len(bird_frames)

        if pipe_x < -PIPE_WIDTH:
            pipe_x = GAME_WIDTH
            pipe_height = random.randint(100, GAME_HEIGHT - PIPE_GAP - 100)
            pipe_passed = False

        if check_collision(bird_x, bird_y, pipe_x, pipe_height):
            game_state = GAME_OVER
            auto_replay_timer = 0  # Reset timer when game over

        bird_right_x = bird_x + BIRD_WIDTH
        pipe_left_x = pipe_x
        bird_center_y = bird_y + BIRD_HEIGHT / 2
        gap_top = pipe_height
        gap_bottom = pipe_height + PIPE_GAP

        if not pipe_passed and bird_right_x > pipe_left_x:
            if gap_top < bird_center_y < gap_bottom:
                score += 1
                pipe_passed = True
                if point_sound:
                    point_sound.play()

        # Clear screen and draw background for game area
        screen.fill(BLACK)
        screen.blit(background, (0, 0))
        draw_pipe(pipe_x, pipe_height)
        draw_bird(bird_x, bird_y, int(bird_frame))
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

    elif game_state == GAME_OVER:
        # Handle auto-replay for AI mode
        if ai_config.use_ai and control_panel.is_auto_replay_enabled():
            auto_replay_timer += 1
            if auto_replay_timer >= AUTO_REPLAY_DELAY:
                reset_game()
                game_state = PLAYING
        
        # Clear screen and draw background for game area
        screen.fill(BLACK)
        screen.blit(background, (0, 0))
        draw_pipe(pipe_x, pipe_height)
        draw_bird(bird_x, bird_y, int(bird_frame))
        game_over_text = render_text_with_outline(
            "Game Over!", font, WHITE, BLACK)
        
        # Show different restart instructions based on auto-replay status
        if ai_config.use_ai and control_panel.is_auto_replay_enabled():
            countdown = (AUTO_REPLAY_DELAY - auto_replay_timer) // 60 + 1
            restart_text = render_text_with_outline(
                f"Auto-restart in {countdown}s", font, WHITE, BLACK)
        else:
            restart_text = render_text_with_outline(
                "Press SPACE to Restart", font, WHITE, BLACK)
        
        game_over_rect = game_over_text.get_rect(
            center=(GAME_WIDTH // 2, GAME_HEIGHT // 2 - 20))
        restart_rect = restart_text.get_rect(
            center=(GAME_WIDTH // 2, GAME_HEIGHT // 2 + 20))
        screen.blit(game_over_text, game_over_rect)
        screen.blit(restart_text, restart_rect)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
    # Draw AI status overlay
    draw_ai_status()
    
    # Draw divider line between game and GUI
    pygame.draw.line(screen, WHITE, (GAME_WIDTH, 0), (GAME_WIDTH, SCREEN_HEIGHT), 2)
    
    # Draw GUI control panel
    control_panel.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

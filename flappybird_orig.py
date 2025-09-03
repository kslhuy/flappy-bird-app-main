import pygame
import random

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

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
        background, (SCREEN_WIDTH, SCREEN_HEIGHT))
else:
    print("Background failed to load, using fallback")
    background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
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
bird_x = SCREEN_WIDTH // 4
bird_y = SCREEN_HEIGHT // 2
bird_velocity = 0
GRAVITY = 0.5
FLAP_POWER = -10
bird_frame = 0
bird_animation_speed = 0.1
bird_animation_counter = 0

# Pipe properties
PIPE_GAP = 150
pipe_x = SCREEN_WIDTH
pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
pipe_speed = 3
pipe_passed = False

# Game variables
score = 0
font = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

# Game states
IDLE = 0
PLAYING = 1
GAME_OVER = 2
game_state = IDLE


def draw_bird(x, y, frame):
    screen.blit(bird_frames[frame], (x, y))


def draw_pipe(x, height):
    top_pipe_height = height
    top_pipe = pygame.transform.flip(pipe_base_img, False, True)
    top_pipe = pygame.transform.scale(top_pipe, (PIPE_WIDTH, top_pipe_height))
    screen.blit(top_pipe, (x, 0))

    bottom_pipe_height = SCREEN_HEIGHT - (height + PIPE_GAP)
    bottom_pipe = pygame.transform.scale(
        pipe_base_img, (PIPE_WIDTH, bottom_pipe_height))
    screen.blit(bottom_pipe, (x, height + PIPE_GAP))


def check_collision(bird_x, bird_y, pipe_x, pipe_height):
    bird_rect = pygame.Rect(bird_x, bird_y, BIRD_WIDTH, BIRD_HEIGHT)
    top_pipe_rect = pygame.Rect(pipe_x, 0, PIPE_WIDTH, pipe_height)
    bottom_pipe_rect = pygame.Rect(
        pipe_x, pipe_height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - (pipe_height + PIPE_GAP))
    if bird_y < 0 or bird_y + BIRD_HEIGHT > SCREEN_HEIGHT:
        if die_sound and (bird_y < 0 or bird_y + BIRD_HEIGHT > SCREEN_HEIGHT):
            die_sound.play()
        return True
    if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
        if hit_sound:
            hit_sound.play()
        return True
    return False


def reset_game():
    global bird_y, bird_velocity, pipe_x, pipe_height, score, pipe_passed, bird_frame
    bird_y = SCREEN_HEIGHT // 2
    bird_velocity = 0
    pipe_x = SCREEN_WIDTH
    pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
    score = 0
    pipe_passed = False
    bird_frame = 0


def render_text_with_outline(text, font, color, outline_color):
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, outline_color)
    surface = pygame.Surface(
        (text_surface.get_width() + 4, text_surface.get_height() + 4), pygame.SRCALPHA)
    for dx, dy in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]:
        surface.blit(outline_surface, (dx + 2, dy + 2))
    surface.blit(text_surface, (2, 2))
    return surface


# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
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

    if game_state == IDLE:
        screen.blit(background, (0, 0))
        bird_y = SCREEN_HEIGHT // 2 + 20 * \
            pygame.math.Vector2(0, 1).rotate(pygame.time.get_ticks() / 100).y
        draw_bird(bird_x, bird_y, int(bird_frame))
        bird_animation_counter += bird_animation_speed
        if bird_animation_counter >= 1:
            bird_animation_counter = 0
            bird_frame = (bird_frame + 1) % len(bird_frames)
        start_text = render_text_with_outline(
            "Press SPACE to Start", font, WHITE, BLACK)
        start_rect = start_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(start_text, start_rect)

    elif game_state == PLAYING:
        bird_velocity += GRAVITY
        bird_y += bird_velocity
        pipe_x -= pipe_speed

        bird_animation_counter += bird_animation_speed
        if bird_animation_counter >= 1:
            bird_animation_counter = 0
            bird_frame = (bird_frame + 1) % len(bird_frames)

        if pipe_x < -PIPE_WIDTH:
            pipe_x = SCREEN_WIDTH
            pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
            pipe_passed = False

        if check_collision(bird_x, bird_y, pipe_x, pipe_height):
            game_state = GAME_OVER

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

        screen.blit(background, (0, 0))
        draw_pipe(pipe_x, pipe_height)
        draw_bird(bird_x, bird_y, int(bird_frame))
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

    elif game_state == GAME_OVER:
        screen.blit(background, (0, 0))
        draw_pipe(pipe_x, pipe_height)
        draw_bird(bird_x, bird_y, int(bird_frame))
        game_over_text = render_text_with_outline(
            "Game Over!", font, WHITE, BLACK)
        restart_text = render_text_with_outline(
            "Press SPACE to Restart", font, WHITE, BLACK)
        game_over_rect = game_over_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
        restart_rect = restart_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        screen.blit(game_over_text, game_over_rect)
        screen.blit(restart_text, restart_rect)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

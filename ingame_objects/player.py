import pygame
import os


class Player(pygame.sprite.Sprite):
    def __init__(self, starting_pos, size_x, size_y, scaling, tiny_vis=False):
        super().__init__()

        # agent animations
        self.animations = {
            'idle': pygame.image.load(os.path.join('assets/spaceship/idle', 'spaceship_master.png')).convert_alpha(),
            'left': pygame.image.load(os.path.join('assets/spaceship/left', 'spaceship_master_left_turn.png')).convert_alpha(),
            'right': pygame.image.load(os.path.join('assets/spaceship/right', 'spaceship_master_right_turn.png')).convert_alpha()
        }
        self.animations['idle'] = pygame.transform.scale(self.animations['idle'], (size_x*scaling, size_y*scaling))
        self.animations['left'] = pygame.transform.scale(self.animations['left'], (size_x*scaling, size_y*scaling))
        self.animations['right'] = pygame.transform.scale(self.animations['right'], (size_x*scaling, size_y*scaling))
        # actual image which will be drawn on player rectangle
        self.image = self.animations['idle']
        # self.image = pygame.image.load(os.path.join('assets/spaceship/idle', 'spaceship_master.png')).convert_alpha()
        # self.image = pygame.transform.scale(self.image, (agent_size_x*scaling, agent_size_y*scaling))
        if tiny_vis:
            self.image = pygame.Surface((size_x*scaling, size_y*scaling))
            self.image.fill('green')
        # rectangular surface of the player
        self.rect = self.image.get_rect(topleft=starting_pos)

        # player movement
        self.direction = pygame.math.Vector2(0, 0)
        # self.velocity = 5  # testing out various player left & right movement velocities
        # imposed movement
        self.drift = pygame.math.Vector2(0, 0)
        self.crashed = False

    def get_input(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            self.direction.x = 1
        elif keys[pygame.K_LEFT]:
            self.direction.x = -1
        else:
            self.direction.x = 0

    def update(self, player_position, scaling, keyboard_input=False):
        if keyboard_input:
            self.get_input()
            player_horizontal_movement = self.direction.x + self.drift.x  # compute horizontal movement with drift
            self.rect.x += player_horizontal_movement * scaling  # apply horizontal drift
        else:
            self.rect = self.image.get_rect(topleft=player_position)

    def animate(self, direction):
        if direction is None:
            self.image = self.animations['idle']
        elif direction == 'Left':
            self.image = self.animations['left']
        elif direction == 'Right':
            self.image = self.animations['right']

    def approach(self, speed, scaling):
        """
        approaching movement of agent at beginning of trial
        """
        self.rect.y += speed*scaling*2/3  # 2/3: artificial decrease in approaching speed

import pygame


class Particle(pygame.sprite.Sprite):
    def __init__(self, pos, size, scaling):
        super().__init__()
        # actual comet image which will be drawn on rectangular surface of comet
        self.image = pygame.Surface((size*scaling, size*scaling))
        self.image.fill('grey')
        self.rect = self.image.get_rect(topleft=pos)

    def update(self, speed, scaling, horizontal_movement):
        # vertical movement
        self.rect.y -= (1*scaling) * speed / 2  # /2 will lead to a distortion effect in which the background moves slower
        # horizontal movement
        self.rect.x += horizontal_movement * scaling * speed  # how to create distortion without particles leaving level bounds

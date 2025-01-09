import pygame


class Line(pygame.sprite.Sprite):
    """
    simple class for visualizing any boarders within the level. A line can be drawn to signal that now input noise will
    be imposed or that the end of the level is approaching or that the first half is done. Room for imagination
    """
    def __init__(self, pos, size, col='darkgrey'):
        super().__init__()
        self.image = pygame.Surface((size[0], size[1]))
        self.image.fill(col)
        self.rect = self.image.get_rect(topleft=pos)

    def update(self, speed, scaling, horizontal_movement):
        # vertical movement
        self.rect.y -= (1*scaling) * speed
        # horizontal movement
        self.rect.x += horizontal_movement * scaling * speed

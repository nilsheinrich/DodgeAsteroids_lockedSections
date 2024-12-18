import pygame
from config import observation_space_size_x, observation_space_size_y, scaling, edge


pygame.init()

# features of displayed text
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

title_font = pygame.font.SysFont('Calibri', 70, bold=True)
text_font = pygame.font.SysFont('Calibri', 35, bold=True)
annotations_font = pygame.font.SysFont('Calibri', 20, bold=True)


def display_instructions(surface):
    surface.fill('black')
    with open("assets/instructions/instructions_text.txt") as f:
        for n, line in enumerate(f):
            instruction_text = text_font.render(line.rstrip('\r\n'), True, WHITE)  # rstrip gets rid of trailing newline characters
            text_rect = instruction_text.get_rect()
            text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
            text_rect.centery = n * 50 + 100
            surface.blit(instruction_text, text_rect)


def display_intertrial_screen(surface):
    surface.fill('black')
    title_lable = title_font.render('Press SPACEBAR to start', 1, WHITE)
    title_rect = title_lable.get_rect()
    title_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
    title_rect.centery = (observation_space_size_y * scaling) // 2
    surface.blit(title_lable, title_rect)


def display_intertrial_screen_after_crash(surface):
    surface.fill('black')
    with open("assets/instructions/crashed_text.txt") as f:
        for n, line in enumerate(f):
            instruction_text = text_font.render(line.rstrip('\r\n'), True, WHITE)
            text_rect = instruction_text.get_rect()
            text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
            text_rect.centery = n * 50 + 100
            surface.blit(instruction_text, text_rect)


def display_soc_question(surface, answered=False):
    surface.fill('black')

    # questionnaire text
    with open("assets/questionnaires/soc_questionnaire_text.txt") as f:
        for n, line in enumerate(f):
            questionnaire_text = text_font.render(line.rstrip('\r\n'), True, WHITE)
            questionnaire_text_rect = questionnaire_text.get_rect()
            questionnaire_text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
            questionnaire_text_rect.centery = n * 50 + 200
            surface.blit(questionnaire_text, questionnaire_text_rect)

    scale_text = "no control   1   2   3   4   5   6   7   full control"
    scale_text = text_font.render(scale_text, True, BLACK, WHITE)
    scale_text_rect = scale_text.get_rect()
    scale_text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
    scale_text_rect.centery = n * 50 + 300
    surface.blit(scale_text, scale_text_rect)

    # questionnaire_instruction text
    with open("assets/questionnaires/soc_questionnaire_instruction.txt") as f:
        for n, line in enumerate(f):
            questionnaire_instr_text = annotations_font.render(line.rstrip('\r\n'), True, WHITE)
            questionnaire_instr_text_rect = questionnaire_instr_text.get_rect()
            questionnaire_instr_text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
            questionnaire_instr_text_rect.centery = n * 50 + 450
            surface.blit(questionnaire_instr_text, questionnaire_instr_text_rect)

    # display continue button when answer provided
    if answered:
        title_lable = title_font.render('Press SPACEBAR to continue', 1, WHITE)
        title_rect = title_lable.get_rect()
        title_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
        title_rect.centery = ((observation_space_size_y * scaling) // 2) + 150
        surface.blit(title_lable, title_rect)


def display_prior_question(surface, answered=False):
    surface.fill('black')

    # questionnaire text
    with open("assets/questionnaires/prior_questionnaire_text.txt") as f:
        for n, line in enumerate(f):
            questionnaire_text = text_font.render(line.rstrip('\r\n'), True, WHITE)
            questionnaire_text_rect = questionnaire_text.get_rect()
            questionnaire_text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
            questionnaire_text_rect.centery = n * 50 + 200
            surface.blit(questionnaire_text, questionnaire_text_rect)

    scale_text = "very poorly   1   2   3   4   5   6   7   excellent"
    scale_text = text_font.render(scale_text, True, BLACK, WHITE)
    scale_text_rect = scale_text.get_rect()
    scale_text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
    scale_text_rect.centery = n * 50 + 300
    surface.blit(scale_text, scale_text_rect)

    # questionnaire_instruction text
    with open("assets/questionnaires/prior_questionnaire_instruction.txt") as f:
        for n, line in enumerate(f):
            questionnaire_instr_text = annotations_font.render(line.rstrip('\r\n'), True, WHITE)
            questionnaire_instr_text_rect = questionnaire_instr_text.get_rect()
            questionnaire_instr_text_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
            questionnaire_instr_text_rect.centery = n * 50 + 450
            surface.blit(questionnaire_instr_text, questionnaire_instr_text_rect)

    # display continue button when answer provided
    if answered:
        title_lable = title_font.render('Press SPACEBAR to continue', 1, WHITE)
        title_rect = title_lable.get_rect()
        title_rect.centerx = (observation_space_size_x * scaling + 2 * edge * scaling) // 2
        title_rect.centery = ((observation_space_size_y * scaling) // 2) + 150
        surface.blit(title_lable, title_rect)

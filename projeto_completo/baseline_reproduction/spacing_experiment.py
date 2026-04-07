def apply_spacing(text, n_spaces):
    """
    Aplica n_spaces caracteres de espaço entre CADA DUAS palavras consecutivas no texto.
    Ex: "palavra1 palavra2 palavra3 palavra4" com n_spaces=5
    se torna "palavra1 palavra2     palavra3 palavra4"
    """
    words = text.split()
    if not words:
        return ""

    # Se houver apenas uma palavra, não há onde inserir espaços
    if len(words) == 1:
        return words[0]

    # Isso significa que o espaço extra é inserido após a segunda palavra, quarta palavra, sexta palavra, etc.
    # Exemplo: "palavra1 palavra2 palavra3 palavra4 palavra5" com n_spaces=5
    # words[0] words[1] (5 espaços) words[2] words[3] (5 espaços) words[4]

    processed_text_parts = []
    for i in range(len(words)):
        processed_text_parts.append(words[i])
        # Se o índice atual for ímpar (ou seja, é a segunda palavra de um par)
        # E não for a última palavra do texto
        if (i % 2 == 1) and (i < len(words) - 1):
            processed_text_parts.append(" " * n_spaces)
        # Se o número total de palavras for ímpar e estamos na penúltima palavra (words[len-2]),
        # não adicionamos espaço extra, pois a última palavra (words[len-1]) não forma um par completo.
        # Ex: p1 p2 (extra) p3 p4 (extra) p5. O extra só vai depois de p2 e p4.
        # Se fosse p1 p2 (extra) p3. O extra só vai depois de p2.
        # A condição (i % 2 == 1) já cuida disso.

    return " ".join(processed_text_parts)

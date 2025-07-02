import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
	modelName: 'gpt-4o',
	temperature: 0.8,
	maxTokens: 700,
});

async function fromTemplate() {
	const prompt = ChatPromptTemplate.fromTemplate('Write a short description for the following product: {product_name}');

	const wholePrompt = await prompt.format({
		product_name: 'iPhone',
	});

	// Creating a chain: connecting the model with the prompt

	const chain = prompt.pipe(model);

	const response = await chain.invoke({
		product_name: 'iPhone',
	});

	console.log(response.content);
}

async function fromMessage() {
	const prompt = ChatPromptTemplate.fromMessages([
		['system', 'You are a helpful assistant that translates {input_lan} to {output_lan}.'],
		['human', 'Translate the following sentence: {sentence}'],
	]);

	const chain = prompt.pipe(model);
	const response = await chain.invoke({
		input_lan: 'English',
		output_lan: 'Hindi',
		sentence:
			'The iPhone is a sleek and powerful smartphone designed by Apple Inc., known for its cutting-edge technology, intuitive user interface, and premium build quality.',
	});

	console.log(response.content);
}

// fromTemplate();
fromMessage();

require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

const DEPLOYER_KEY = process.env.DEPLOYER_PRIVATE_KEY || "0x" + "0".repeat(64);
const ETHERSCAN_KEY = process.env.ETHERSCAN_API_KEY || "";

module.exports = {
  solidity: {
    version: "0.8.24",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      evmVersion: "cancun",
    },
  },
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org",
      accounts: [DEPLOYER_KEY],
    },
    mainnet: {
      url: process.env.MAINNET_RPC_URL || "",
      accounts: [DEPLOYER_KEY],
    },
  },
  etherscan: {
    apiKey: ETHERSCAN_KEY,
  },
};

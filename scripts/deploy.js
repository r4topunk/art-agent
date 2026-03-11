const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const ArtAgent = await hre.ethers.getContractFactory("ArtAgent");
  const contract = await ArtAgent.deploy(deployer.address);
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("ArtAgent deployed to:", address);

  // Save deployment info for the Python minting module
  const fs = require("fs");
  const deployment = {
    address,
    deployer: deployer.address,
    network: hre.network.name,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    deployedAt: new Date().toISOString(),
  };
  fs.writeFileSync(
    "deployment.json",
    JSON.stringify(deployment, null, 2) + "\n"
  );
  console.log("Deployment info saved to deployment.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
